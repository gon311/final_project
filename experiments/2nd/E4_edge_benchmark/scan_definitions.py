"""E4 0단계: 130장(0.3절 기준, 시편 정보 있는 사진)을 디코딩해 베어링부 상·하단과 핀
상·하단에서 5가지 엣지 정의(cross50, grad_peak, moment_centroid, ESF 1~3단, 하단 문턱
10~90% 9단계)를 모두 열별로 뽑는다. 원자료(이미지×열 단위, 10MB 전후)라 `cache/e4/`에
두고(0.6절) 커밋하지 않는다 — 요약은 `step1_summary.py` 이후 스크립트가 `results/`에 낸다.

베어링부는 기본 `--step 10`으로 훑는다(E1의 `step=2`보다 거칠다 — ESF 최소제곱 적합이
열마다 scipy 비선형 solver를 돌려서 느리고, 이 실험은 "정의 간 상대적 안정성"을 보는 게
목적이라 `step=2`만큼 빽빽할 필요가 없다). `--step 2`로 다시 돌리면 `bearing_columns_step2.csv`
에 저장된다 — `step8_step_size_check.py`가 이 둘을 모든 정의(cross50뿐 아니라 moment·ESF
까지)로 비교한다(Codex adversarial-review 반영, README "Codex 리뷰 반영 표" 참고).
핀은 E5가 고른 pin_relative 창(`PIN_RELATIVE_WINDOW`, 폭 45px)이 좁아 항상 `step=2`다.

**ESF 적합 실패(`success=False`, `esf_fit`이 fallback으로 grad-peak를 복제한 값)는 그대로
쓰지 않고 NaN으로 바꾼다**(Codex adversarial-review 반영) — 실패를 "측정값"으로 섞으면
안정성·체계차 판단이 왜곡된다. `*_ok` 열(0/1)에 성공 여부를 같이 남겨 실패율을 따로 볼 수
있게 했다.

**cross50·grad_peak(`edge_subpx` 호출)는 `EDGE_SUBPX_WIN=20`을 쓴다 — `measure.py`
`_measure_window`가 쓰는 창과 똑같다**(Codex `/codex:review` 반영). 처음엔 전부
`WIN=45`(moment·ESF가 다단을 보려고 넓게 잡은 창, E3 `multistep.py`와 같은 값)를 썼는데,
`edge_subpx`는 `[center-win, center+win]` 안에서 가장 강한 그래디언트를 고르기 때문에
창을 넓히면 E1이 원래 고르던 엣지와 다른(더 멀리 있는, 예컨대 반사띠) 지점을 고를 수 있다
— 그러면 이 벤치마크의 "cross50"·"grad_peak"가 더 이상 E1·팀원A·팀원B가 말하는 그 정의와
같지 않게 된다. moment_centroid·esf_fit·level_crossing(하단 문턱 sweep)은 다단 구조를
보려는 목적이 있으므로 계속 `MULTISTEP_WIN=45`를 쓴다 — 정의마다 다른 창을 쓰는 게
의도된 설계다(E1이 "같은 정의"로 보던 걸 왜곡하지 않으면서, 새 정의는 원래 목적대로
넓게 보게 둔다).
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
from shaftlab.measure import BEARING_COARSE, PIN_COARSE, PIN_RELATIVE_WINDOW, edge_subpx  # noqa: E402
from shaftlab.edge_methods import esf_fit, level_crossing, moment_centroid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache", "e4")
META_COLS = ["묶음", "시편", "역할", "등급", "실측mm", "초점", "핀배치"]
PIN_STEP = 2
EDGE_SUBPX_WIN = 20  # measure.py _measure_window와 동일 — cross50/grad_peak(edge_subpx) 전용
MULTISTEP_WIN = 45   # E3 multistep.py와 동일 — moment/ESF/문턱 sweep처럼 다단을 보는 정의 전용
THRESH_LEVELS = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]


def column_profile(full, seg, X, half=80):
    si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
    tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
    y0, y1 = tc - half, bc + half
    prof = full[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
    return prof, y0, tc - y0, bc - y0


def _esf_centers_or_nan(prof, center, win, sign, n_steps):
    """`esf_fit` 결과에서 성공한 경우만 중심 위치를 쓴다. 실패하면(비수렴·초기 peak 부족)
    fallback(grad-peak 복제값)을 그대로 쓰지 않고 NaN n_steps개 + ok=False를 반환한다."""
    r = esf_fit(prof, center, win, sign, n_steps=n_steps)
    if r["success"]:
        return r["centers"], True
    return [np.nan] * n_steps, False


def scan_bearing(gray, meta_dict, fn, seg, step):
    xa, xb = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
    m = int((xb - xa) * 0.2)
    xa, xb = xa + m, xb - m
    rows = []
    for X in range(xa, xb, step):
        prof, y0, tc, bc = column_profile(gray, seg, X)
        t_peak, t_cross, *_ = edge_subpx(prof, tc, EDGE_SUBPX_WIN, -1)
        b_peak, b_cross, *_ = edge_subpx(prof, bc, EDGE_SUBPX_WIN, +1)
        t_mom = moment_centroid(prof, tc, MULTISTEP_WIN, -1)
        b_mom = moment_centroid(prof, bc, MULTISTEP_WIN, +1)
        t_esf1, t_esf1_ok = _esf_centers_or_nan(prof, tc, MULTISTEP_WIN, -1, 1)
        b_esf1, b_esf1_ok = _esf_centers_or_nan(prof, bc, MULTISTEP_WIN, +1, 1)
        b_esf2, b_esf2_ok = _esf_centers_or_nan(prof, bc, MULTISTEP_WIN, +1, 2)
        b_esf3, b_esf3_ok = _esf_centers_or_nan(prof, bc, MULTISTEP_WIN, +1, 3)
        row = dict(file=fn, **meta_dict, x=X,
                   top_cross=t_cross + y0, top_peak=t_peak + y0, top_moment=t_mom + y0,
                   top_esf1=t_esf1[0] + y0, top_esf1_ok=t_esf1_ok,
                   bot_cross=b_cross + y0, bot_peak=b_peak + y0, bot_moment=b_mom + y0,
                   bot_esf1=b_esf1[0] + y0, bot_esf1_ok=b_esf1_ok,
                   bot_esf2_inner=b_esf2[0] + y0, bot_esf2_outer=b_esf2[1] + y0, bot_esf2_ok=b_esf2_ok,
                   bot_esf3_inner=b_esf3[0] + y0, bot_esf3_mid=b_esf3[1] + y0, bot_esf3_outer=b_esf3[2] + y0,
                   bot_esf3_ok=b_esf3_ok)
        for lv in THRESH_LEVELS:
            row[f"bot_thresh{int(lv * 100)}"] = level_crossing(prof, bc, MULTISTEP_WIN, +1, level=lv) + y0
        rows.append(row)
    return rows


def scan_pin(gray, meta_dict, fn, seg):
    xL, xR = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
    xa = xL + PIN_RELATIVE_WINDOW[0]
    xb = min(xL + PIN_RELATIVE_WINDOW[1], xR)
    if xb - xa < 3 * PIN_STEP:
        return []
    rows = []
    for X in range(xa, xb, PIN_STEP):
        prof, y0, tc, bc = column_profile(gray, seg, X)
        t_peak, t_cross, *_ = edge_subpx(prof, tc, EDGE_SUBPX_WIN, -1)
        b_peak, b_cross, *_ = edge_subpx(prof, bc, EDGE_SUBPX_WIN, +1)
        t_mom = moment_centroid(prof, tc, MULTISTEP_WIN, -1)
        b_mom = moment_centroid(prof, bc, MULTISTEP_WIN, +1)
        t_esf1, t_esf1_ok = _esf_centers_or_nan(prof, tc, MULTISTEP_WIN, -1, 1)
        b_esf1, b_esf1_ok = _esf_centers_or_nan(prof, bc, MULTISTEP_WIN, +1, 1)
        rows.append(dict(file=fn, **meta_dict, x=X, xL=xL, xR=xR,
                          top_cross=t_cross + y0, top_peak=t_peak + y0, top_moment=t_mom + y0,
                          top_esf1=t_esf1[0] + y0, top_esf1_ok=t_esf1_ok,
                          bot_cross=b_cross + y0, bot_peak=b_peak + y0, bot_moment=b_mom + y0,
                          bot_esf1=b_esf1[0] + y0, bot_esf1_ok=b_esf1_ok))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10, help="베어링부 열 스캔 간격(px). 기본 10.")
    args = ap.parse_args()

    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]
    files = sorted(meta.index)

    bearing_rows, pin_rows = [], []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        m = meta.loc[fn]
        meta_dict = {c: m[c] for c in META_COLS}

        seg_b = coarse_section(q, **BEARING_COARSE)
        bearing_rows += scan_bearing(gray, meta_dict, fn, seg_b, args.step)

        if args.step == 10:  # 핀은 step=2 고정 산출물 하나만 있으면 된다(step 교차검증 대상 아님)
            seg_p = coarse_section(q, **PIN_COARSE)
            if len(seg_p) >= 6:
                pin_rows += scan_pin(gray, meta_dict, fn, seg_p)

        print(k, fn, round(time.time() - t0, 2), flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    bearing_name = "bearing_columns.csv" if args.step == 10 else f"bearing_columns_step{args.step}.csv"
    out_sets = [(bearing_rows, bearing_name)]
    if args.step == 10:
        out_sets.append((pin_rows, "pin_columns.csv"))
    for rows, name in out_sets:
        df = pd.DataFrame(rows)
        path = os.path.join(CACHE_DIR, name)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path} ({len(df)}행)")


if __name__ == "__main__":
    main()
