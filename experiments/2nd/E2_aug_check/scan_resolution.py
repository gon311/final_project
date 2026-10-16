"""E2 4단계(해상도 조건, 증강 판정과 별개): 2x2·4x4 화소 평균 축소(`augment.pixel_average_downsample`,
원본 크기로 다시 늘려서 좌표 상수는 그대로 쓴다) + 2x2의 격자 위상 4가지에서 Δ외경과 하단
단 전환을 본다.

**이 조건만 `coarse_section`을 매번 새로 돌린다**(증강 비교, `scan_aug_effect.py`와의
차이) — "좌표 상수(`BEARING_COARSE`)가 해상도가 달라져도 셔프트를 그대로 찾는가"가 이
조건이 묻는 질문 자체라서다. 새로 찾은 열에서 쟀으니, Δ는 "같은 자리를 다르게 쟀다"와
"어디를 쟀는지부터 달라졌다"가 섞여 있을 수 있다 — 해석할 때 유념한다(README "한계").

**130장 전체가 기본이다**(Codex adversarial-review 반영, `scan_aug_effect.py`와 같은
이유). `--sample`로 예전 26장만 빠르게 돌릴 수 있다.

**단 전환 비교는 "같은 상대 위치"(0~100% 등분 인덱스)끼리만 한다 — 모호한 열을 걸러낸
뒤 리스트 순서로 비교하지 않는다**(Codex `/codex:review` 반영). 전에는
`bottom_step_samples`가 모호한 열을 아예 빼고 돌려줘서, 조건마다 모호한 열이 다르면
(블러·축소 조건에서 실제로 흔하다) 걸러낸 뒤 순서로 짝지을 때 서로 다른 물리적 열끼리
비교되는 버그가 있었다. 이제 길이를 항상 고정하고 모호한 자리는 `None`으로 채워서,
같은 인덱스(같은 상대 위치)만 비교한다.

입력: 원본 이미지(paths.IMAGE_DIR), paths.EDA_MEASURE_RAW_CSV(이미지 선정·메타데이터)
산출물: results/resolution_check.csv
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, measure_bearing  # noqa: E402
from shaftlab.multistep import edge_multistep_features  # noqa: E402
import augment as A  # noqa: E402
from scan_aug_effect import MULTISTEP_WIN, pick_all_files, pick_representative_files  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = [("2x2", 2, (0, 0)), ("2x2", 2, (1, 0)), ("2x2", 2, (0, 1)), ("2x2", 2, (1, 1)),
              ("4x4", 4, (0, 0))]


def bottom_step_samples(gray: np.ndarray, seg: np.ndarray, n_samples: int = 8) -> list:
    """셋업마다(조건마다) 자기 탐지 구간을 0~100% 상대 위치로 `n_samples`등분해 샘플링한다.
    **길이가 항상 `n_samples`로 고정**이고, 모호한 열은 버리지 않고 `None`으로 채운다
    (Codex `/codex:review` 반영) — 전에는 모호한 열을 걸러낸 뒤 남은 걸 리스트 순서로
    비교해서, 조건마다 어느 열이 모호한지가 다르면(블러·축소 조건에서 실제로 흔하다)
    서로 다른 물리적 열끼리 비교하는 버그가 있었다. 이제 호출부가 **같은 인덱스 =
    같은 상대 위치**로만 비교하므로 정렬이 깨지지 않는다."""
    xa, xb = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
    m = int((xb - xa) * 0.2)
    xa, xb = xa + m, xb - m
    if xb <= xa:
        return [None] * n_samples
    xs = np.linspace(xa, xb - 1, min(n_samples, max(1, xb - xa))).astype(int)
    steps = [None] * n_samples
    for i, X in enumerate(xs):
        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        feat = edge_multistep_features(prof, bc - y0, MULTISTEP_WIN, +1, n_peaks=3)
        if not feat["cross50_step_ambiguous"]:
            steps[i] = feat["cross50_step"]
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="26장만 빠르게 — 기본은 130장 전체")
    args = ap.parse_args()
    files = pick_representative_files() if args.sample else pick_all_files()
    out_name = "resolution_check_sample26.csv" if args.sample else "resolution_check.csv"
    print(f"이미지 {len(files)}장, 해상도 조건 {len(CONDITIONS)}종 ({'표본' if args.sample else '전체'})")

    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")

    rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        try:
            base = measure_bearing(gray)
        except ValueError as e:
            print(f"  (경고) {fn} 원본 측정 실패: {e}")
            continue
        q0 = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        seg0 = coarse_section(q0, **BEARING_COARSE)
        base_steps = bottom_step_samples(gray.astype(np.float64), seg0)

        for name, n, phase in CONDITIONS:
            ds = A.pixel_average_downsample(gray, n, phase)
            try:
                r = measure_bearing(ds)
            except ValueError as e:
                rows.append(dict(file=fn, 묶음=int(meta.loc[fn, "묶음"]), 시편=meta.loc[fn, "시편"],
                                  condition=name, phase=str(phase), ok=False, error=str(e)))
                continue
            q = cv2.resize(ds, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            seg = coarse_section(q, **BEARING_COARSE)
            ds_steps = bottom_step_samples(ds, seg)
            # 같은 인덱스(= 같은 상대 위치, 0~100% 등분)끼리만 비교한다 — 둘 다 None이
            # 아닌 경우만 유효 비교로 센다(Codex `/codex:review` 반영, 위 함수 docstring
            # 참고. 리스트 길이가 항상 같아 position 밀림이 없다).
            pairs = [(a, b) for a, b in zip(base_steps, ds_steps) if a is not None and b is not None]
            trans_frac = float(np.mean([a != b for a, b in pairs])) if pairs else float("nan")
            rows.append(dict(
                file=fn, 묶음=int(meta.loc[fn, "묶음"]), 시편=meta.loc[fn, "시편"],
                condition=name, phase=str(phase), ok=True,
                d_diam_cross50_px=r["d_line"] - base["d_line"],
                d_diam_grad_peak_px=r["d_peak_median"] - base["d_peak_median"],
                n_cols_detected=r["n_cols"], base_n_cols_detected=base["n_cols"],
                bot_step_transition_frac=trans_frac,
            ))
        print(k, fn, round(time.time() - t0, 2), flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", out_name)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(out)}행)")


if __name__ == "__main__":
    main()
