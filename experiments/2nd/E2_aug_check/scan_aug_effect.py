"""E2 1단계: 130장(0.3절 기준, 시편 정보 있는 사진) 전체에 증강을 적용하고, 원본 대비
Δ외경·Δ상단·Δ하단을 cross50·grad_peak 정의별로 잰다. 하단은 단 전환(E3 정의)도 같이 센다.

**처음엔 "셋업별 대표 2장"(묶음당 시각순 첫·마지막, 26장)만 돌렸는데, Codex
adversarial-review가 "26장 중앙값으로 허용/불허를 확정하면 특정 셋업·프레임 상태에서만
나오는 최악 조건이 가려질 수 있다"고 지적해 130장 전체로 넓혔다.** `--sample`을 주면
예전처럼 26장만(개발·디버깅용, 약 5분) 돌릴 수 있다 — 기본은 전체(약 25~35분).

**Δ외경은 `_measure_window`(E1·`measure_bearing`이 실제로 쓰는 강건 직선적합 +
`cos(기울기)` 보정)로 잰다 — 고정 열에서 상·하단 위치를 단순히 뺀 값이 아니다**(Codex
`/codex:review` 반영). 회전한 원기둥은 물리적 지름이 그대로여도 고정된 X열에서 본 수직
거리가 `1/cos(θ)`만큼 커진다 — 이 베어링(~545px)에서 1°면 약 0.083px로, 처음 버전이
"회전으로 흔들리는 외경"이라 보고한 잔차(±1°에서 0.05~0.09px)와 거의 같은 크기였다.
즉 그 잔차의 상당 부분이 측정 불안정성이 아니라 **코사인 보정을 안 해서 생긴, 처음부터
예측 가능한 기하학적 편향**이었다. cross50 외경(`d_line`)은 코사인 보정까지 그대로
반영되고, grad_peak 외경(`d_peak_median`)은 원래 프로젝트 정의 자체가 열별 차이의
중앙값이라 코사인 보정이 없다 — `_measure_window`를 그대로 가져다 썼으니 두 정의 다
기존 시스템(E1)이 실제로 쓰는 식과 정확히 같다. Δ상단·Δ하단(개별 엣지 위치)은 여전히
보정 없는 원래 값을 쓴다 — 회전으로 실제 위치가 크게 움직이는 걸 보려는 목적이라 보정이
없는 쪽이 맞다.

**같은 열(원본의 `coarse_section` 결과)을 증강 전후 공통으로 쓴다** — 증강마다
`coarse_section`을 다시 돌리면(특히 밝기·대비 변화에서 1/4 축소 미리보기의 110 문턱이
걸리는 열 범위 자체가 바뀔 수 있어) "같은 자리를 쟀는데 값이 얼마나 바뀌었나"가 아니라
"어디를 쟀는지 자체가 바뀐 효과"까지 섞인다. 회전(최대 1°)은 측정 폭(~245px)에 걸쳐
세로로 최대 ~4.3px만 밀리는데, 이는 `edge_subpx`의 탐색 창(±20px)과 열 추출 반창(80px)
안에 충분히 들어오므로 같은 열을 재사용해도 엣지를 놓치지 않는다(코드로 확인함).
해상도(화소 평균 축소) 조건은 성격이 달라 `scan_resolution.py`에서 **매번 새로` coarse_section`
을 돌린다**(그게 이 조건의 질문이다 — "좌표 상수가 해상도 바뀌어도 그대로 맞는가").

입력: 원본 이미지(paths.IMAGE_DIR), paths.EDA_MEASURE_RAW_CSV(대표 이미지 선정용)
산출물: results/aug_effect.csv
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
from shaftlab.measure import BEARING_COARSE, _measure_window, edge_subpx  # noqa: E402
from shaftlab.multistep import edge_multistep_features  # noqa: E402
import augment as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_SUBPX_WIN = 20      # measure.py _measure_window와 동일(cross50/grad_peak)
MULTISTEP_WIN = 45       # E3/E4와 동일(단 전환 판정용)
COL_STEP = 20            # 대표 열 간격(px) — 전수 스캔이 아니라 대표 표본

AUGS = []
for s in (0.5, 1, 2):
    AUGS.append(("blur_gaussian", s, lambda im, s=s: A.blur(im, s)))
for d in (-40, -20, -10, 10, 20, 40):
    AUGS.append(("brightness", d, lambda im, d=d: A.brightness(im, d)))
for f in (0.8, 1.2):
    AUGS.append(("contrast", f, lambda im, f=f: A.contrast(im, f)))
for g in (0.8, 1.25):
    AUGS.append(("gamma", g, lambda im, g=g: A.gamma(im, g)))
for deg in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
    AUGS.append(("rotation_deg", deg, lambda im, deg=deg: A.rotate(im, deg)))
for dy in (0.25, 0.5):
    AUGS.append(("subpixel_shift_y_px", dy, lambda im, dy=dy: A.subpixel_shift_y(im, dy)))


def pick_representative_files() -> list:
    """예전 26장(묶음당 시각순 첫·마지막) 샘플 — `--sample`로만 쓴다."""
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig")
    meta = meta[meta["시편"].notna()]
    files = []
    for _, sub in meta.groupby("묶음"):
        sub = sub.sort_values("시각")
        files.append(sub.iloc[0]["file"])
        files.append(sub.iloc[-1]["file"])
    return files


def pick_all_files() -> list:
    """0.3절 기준 130장 전부(시편 정보 있는 사진) — 기본값(Codex adversarial-review 반영)."""
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig")
    meta = meta[meta["시편"].notna()]
    return sorted(meta["file"])


def measure_columns(gray: np.ndarray, seg: np.ndarray, xs: list) -> pd.DataFrame:
    """대표 열 `xs`에서 cross50·grad_peak 상·하단 위치와 하단 cross50_step(단 번호)을 뽑는다."""
    rows = []
    for X in xs:
        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        t_peak, t_cross, *_ = edge_subpx(prof, tc - y0, EDGE_SUBPX_WIN, -1)
        b_peak, b_cross, *_ = edge_subpx(prof, bc - y0, EDGE_SUBPX_WIN, +1)
        bot_feat = edge_multistep_features(prof, bc - y0, MULTISTEP_WIN, +1, n_peaks=3)
        rows.append(dict(x=X, top_cross=t_cross + y0, bot_cross=b_cross + y0,
                          top_peak=t_peak + y0, bot_peak=b_peak + y0,
                          bot_cross50_step=bot_feat["cross50_step"],
                          bot_step_ambiguous=bot_feat["cross50_step_ambiguous"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                     help="26장(묶음당 첫·마지막)만 빠르게 — 기본은 130장 전체")
    args = ap.parse_args()
    files = pick_representative_files() if args.sample else pick_all_files()
    out_name = "aug_effect_sample26.csv" if args.sample else "aug_effect.csv"
    print(f"이미지 {len(files)}장, 증강 {len(AUGS)}종 ({'표본' if args.sample else '전체'})")

    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")

    rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        seg = coarse_section(q, **BEARING_COARSE)
        xa, xb = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
        m = int((xb - xa) * 0.2)
        xa, xb = xa + m, xb - m
        xs = list(range(xa, xb, COL_STEP))

        base = measure_columns(gray.astype(np.float64), seg, xs)
        try:
            base_w = _measure_window(gray.astype(np.float64), seg, xa, xb, step=COL_STEP)
        except TypeError:
            # `_measure_window`(E1, 수정 안 함)의 MAD 강건적합이 열이 ~7~8개뿐일 때
            # 드물게 전부를 이상치로 쳐 내는 퇴화 사례가 있다(코드로 확인: 대부분 점이
            # 거의 한 직선 위에 있어 MAD≈0이 되는 경우). 이 열에서 d_line/d_peak_median을
            # 못 구하니 그 이미지의 Δ외경만 NaN으로 남기고(Δ상단·하단·단전환은 그대로
            # 쓴다), 다음 이미지로 넘어간다.
            base_w = None
            print(f"  (경고) {fn}: _measure_window 강건적합 실패, 이 이미지는 Δ외경 NaN")

        for aug_name, intensity, fn_aug in AUGS:
            aug_img = fn_aug(gray)
            cur = measure_columns(aug_img, seg, xs)

            d_top_cross = float((cur["top_cross"] - base["top_cross"]).median())
            d_bot_cross = float((cur["bot_cross"] - base["bot_cross"]).median())
            d_top_peak = float((cur["top_peak"] - base["top_peak"]).median())
            d_bot_peak = float((cur["bot_peak"] - base["bot_peak"]).median())
            step_changed = (cur["bot_cross50_step"] != base["bot_cross50_step"])
            ambiguous_either = base["bot_step_ambiguous"] | cur["bot_step_ambiguous"]

            if base_w is not None:
                try:
                    cur_w = _measure_window(aug_img, seg, xa, xb, step=COL_STEP)
                    d_diam_cross = cur_w["d_line"] - base_w["d_line"]
                    d_diam_peak = cur_w["d_peak_median"] - base_w["d_peak_median"]
                except TypeError:
                    d_diam_cross = d_diam_peak = float("nan")
            else:
                d_diam_cross = d_diam_peak = float("nan")

            rows.append(dict(
                file=fn, 묶음=int(meta.loc[fn, "묶음"]), 시편=meta.loc[fn, "시편"],
                aug=aug_name, intensity=intensity,
                d_top_cross50_px=d_top_cross, d_bot_cross50_px=d_bot_cross,
                # Δ외경은 _measure_window의 d_line(cross50, 직선적합+코사인 보정)과
                # d_peak_median(grad_peak, E1 원래 정의대로 보정 없음)으로 잰다 — 고정 열의
                # 상·하단 차이를 그냥 빼면 회전에서 1/cos(θ) 기하 편향이 섞인다(Codex
                # `/codex:review` 반영, 위 모듈 docstring 참고).
                d_diam_cross50_px=d_diam_cross,
                d_top_grad_peak_px=d_top_peak, d_bot_grad_peak_px=d_bot_peak,
                d_diam_grad_peak_px=d_diam_peak,
                n_cols=len(xs), bot_step_transition_frac=float(step_changed.mean()),
                bot_step_transition_frac_확실=float((step_changed & ~ambiguous_either).mean()),
            ))

        print(k, fn, round(time.time() - t0, 2), flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", out_name)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(out)}행)")


if __name__ == "__main__":
    main()
