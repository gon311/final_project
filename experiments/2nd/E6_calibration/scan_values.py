"""E6 0단계: 캘리브레이션 비교에 쓸 베어링부·핀 px를 셋 정의(cross50, grad_peak,
moment_centroid — "E4 최선")로 모은다.

- **베어링부 cross50·grad_peak는 새로 재지 않고 `eda_measure_raw.csv`(E1이 재현한 원본,
  step=2)의 `B_d_line`/`B_d_peak_median`을 그대로 쓴다** — 다시 계산하면 E4처럼 거친
  step(10)을 또 쓰게 되거나 중복 구현이 생긴다. 이 값이 "그 정의"의 공식 기준이다.
- **핀 cross50·grad_peak는 `measure_pin_relative()`(E5가 고른 기본값)로 새로 잰다** —
  기존 `pin_gauge_raw.csv`는 E1 재현 당시의 "auto" 위치 방식 기준이라 쓸 수 없다(E5에서
  기본값이 pin_relative로 바뀜, measure.py 참고).
- **moment_centroid(E4 "최선" 두 번째 후보)는 베어링부·핀 둘 다 새로 잰다** — 기존 CSV에
  없는 정의라서다. `shaftlab.edge_methods.moment_centroid`를 그대로 쓰고
  (`MULTISTEP_WIN=45`, E4와 같은 창), 열별 지름의 중앙값을 이미지당 값으로 쓴다
  (E4 `step1_summary.py`의 `diameter_px` 집계와 같은 방식).

입력: paths.EDA_MEASURE_RAW_CSV(묶음·시편·실측mm·초점·핀배치·시각·cross50·grad_peak),
원본 이미지(paths.IMAGE_DIR).
산출물: results/values_by_image.csv (130행 — 0.3절 기준 시편 있는 사진).
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, PIN_COARSE, PIN_RELATIVE_WINDOW, measure_pin_relative  # noqa: E402
from shaftlab.edge_methods import moment_centroid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
META_COLS = ["묶음", "시편", "역할", "등급", "실측mm", "초점", "핀배치", "시각"]
MULTISTEP_WIN = 45  # E4와 동일 — moment_centroid 전용 창


def moment_diameter(gray, seg, xa, xb, step):
    """[xa, xb) 구간을 `step` 간격으로 훑어 열별 moment_centroid 지름의 중앙값을 낸다.
    `_measure_window`와 같은 열 프로파일 추출 방식(80px 반창, (1,7) 가우시안 블러)을 쓴다."""
    diffs = []
    for X in range(xa, xb, step):
        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        t = moment_centroid(prof, tc - y0, MULTISTEP_WIN, -1)
        b = moment_centroid(prof, bc - y0, MULTISTEP_WIN, +1)
        diffs.append(b - t)
    return float(np.median(diffs)) if diffs else float("nan")


def main():
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]
    files = sorted(meta.index)

    rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        m = meta.loc[fn]
        row = {c: m[c] for c in META_COLS}
        row["file"] = fn
        row["bearing_cross50_px"] = float(m["B_d_line"])
        row["bearing_grad_peak_px"] = float(m["B_d_peak_median"])

        seg_b = coarse_section(q, **BEARING_COARSE)
        xa, xb = int(seg_b[0, 0] * 4), int(seg_b[-1, 0] * 4)
        mrg = int((xb - xa) * 0.2)
        row["bearing_moment_px"] = moment_diameter(gray, seg_b, xa + mrg, xb - mrg, step=4)

        pr = measure_pin_relative(gray)
        row["pin_cross50_px"] = pr["d_line"]
        row["pin_grad_peak_px"] = pr["d_peak_median"]
        row["pin_window_clipped"] = pr["window_clipped"]
        row["pin_xL"] = pr["xL"]

        seg_p = coarse_section(q, **PIN_COARSE)
        if len(seg_p) >= 6:
            xL, xR = int(seg_p[0, 0] * 4), int(seg_p[-1, 0] * 4)
            pa = xL + PIN_RELATIVE_WINDOW[0]
            pb = min(xL + PIN_RELATIVE_WINDOW[1], xR)
            row["pin_moment_px"] = moment_diameter(gray, seg_p, pa, pb, step=2) if pb - pa >= 6 else float("nan")
        else:
            row["pin_moment_px"] = float("nan")

        rows.append(row)
        print(k, fn, round(time.time() - t0, 2), flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "values_by_image.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(out)}행)")


if __name__ == "__main__":
    main()
