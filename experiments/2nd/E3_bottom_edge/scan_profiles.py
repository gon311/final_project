"""E3 0단계: 130장(분류표에 시편 정보가 있는 이미지, 0.3절) 전체를 디코딩해 베어링부
상·하단, 핀 상·하단, 우측 직선부(R_) 하단의 열별 다단 엣지 특징(gradient 극대 상위 3개,
cross50, 배경·안쪽 밝기, 엣지 폭)을 뽑는다.

입력: 원본 이미지(paths.IMAGE_DIR), paths.EDA_MEASURE_RAW_CSV(분류표 메타데이터)
산출물: results/bottom_peaks.csv(베어링부 하단, 1단계 주 산출물), results/top_peaks.csv
       (베어링부 상단, 5단계 대칭성용), results/pin_edge_peaks.csv(핀 상·하단, 4단계),
       results/r_bottom_peaks.csv(우측 직선부 하단, 6단계)
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, PIN_COARSE, R_COARSE  # noqa: E402
from shaftlab.multistep import scan_section_multistep  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
META_COLS = ["묶음", "시편", "역할", "등급", "실측mm", "초점", "핀배치"]
N_PEAKS = 3


def flatten_side(d: dict, prefix: str) -> dict:
    """edge_multistep_features() 결과를 CSV 한 행에 쓸 수 있게 평탄화한다."""
    out = {
        f"{prefix}n_peaks": d["n_peaks_detected"],
        f"{prefix}cross50": d["cross50"],
        f"{prefix}cross50_step": d["cross50_step"],
        f"{prefix}cross50_step_ambiguous": d["cross50_step_ambiguous"],
        f"{prefix}bg_level": d["bg_level"],
        f"{prefix}fg_level": d["fg_level"],
        f"{prefix}inner_ratio": (d["fg_level"] / d["bg_level"]) if d["bg_level"] else np.nan,
        f"{prefix}edge_width_10_90": d["edge_width_10_90"],
        f"{prefix}edge_width_90_to_50": d["edge_width_90_to_50"],
    }
    for k in range(N_PEAKS):
        if k < len(d["peaks"]):
            pos, strength = d["peaks"][k]
        else:
            pos, strength = np.nan, np.nan
        out[f"{prefix}peak{k + 1}_y"] = pos
        out[f"{prefix}peak{k + 1}_strength"] = strength
    return out


def main():
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]  # 0.3절(메모 사진 제외), E1/E5와 같은 기준
    files = sorted(meta.index)

    bottom_rows, top_rows, pin_rows, r_rows = [], [], [], []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        m = meta.loc[fn]
        meta_dict = {c: m[c] for c in META_COLS}

        # 베어링부: 상·하단 둘 다(5단계 대칭성 비교용)
        seg_b = coarse_section(q, **BEARING_COARSE)
        xa_b, xb_b = int(seg_b[0, 0] * 4), int(seg_b[-1, 0] * 4)
        for row in scan_section_multistep(gray, seg_b, xa_b, xb_b, step=2, half_window=45, n_peaks=N_PEAKS):
            base = dict(file=fn, **meta_dict, x=row["x"])
            bottom_rows.append({**base, **flatten_side(row["bot"], "")})
            top_rows.append({**base, **flatten_side(row["top"], "")})

        # 핀: 상·하단 둘 다(4단계)
        seg_p = coarse_section(q, **PIN_COARSE)
        if len(seg_p) >= 6:
            xa_p, xb_p = int(seg_p[0, 0] * 4), int(seg_p[-1, 0] * 4)
            for row in scan_section_multistep(gray, seg_p, xa_p, xb_p, step=2, half_window=45, n_peaks=N_PEAKS):
                base = dict(file=fn, **meta_dict, x=row["x"])
                pin_rows.append({**base, "side": "top", **flatten_side(row["top"], "")})
                pin_rows.append({**base, "side": "bot", **flatten_side(row["bot"], "")})

        # 우측 직선부(R_): 하단만(6단계)
        seg_r = coarse_section(q, **R_COARSE)
        if len(seg_r) >= 6:
            xa_r, xb_r = int(seg_r[0, 0] * 4), int(seg_r[-1, 0] * 4)
            for row in scan_section_multistep(gray, seg_r, xa_r, xb_r, step=4, half_window=45, n_peaks=N_PEAKS):
                base = dict(file=fn, **meta_dict, x=row["x"])
                r_rows.append({**base, **flatten_side(row["bot"], "")})

        print(k, fn, round(time.time() - t0, 2), flush=True)

    def save(rows, out_dir, name):
        df = pd.DataFrame(rows)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path} ({len(df)}행)")

    results_dir = os.path.join(HERE, "results")
    cache_dir = os.path.join(HERE, "..", "cache", "e3")  # 이미지x열 단위 대용량 원자료, 커밋 안 함
    # bottom_peaks.csv만 지시서가 명시한 산출물(이미지x열)이라 results/에 둔다(3.8MB, 10MB 이하).
    # top/pin/R_ 원자료는 이후 요약 스크립트가 cache/에서 읽어 작은 요약 CSV만 results/에 낸다
    # (0.6절: 이미지x열 단위 대용량은 cache/에, 요약만 커밋).
    save(bottom_rows, results_dir, "bottom_peaks.csv")
    save(top_rows, cache_dir, "top_peaks.csv")
    save(pin_rows, cache_dir, "pin_edge_peaks.csv")
    save(r_rows, cache_dir, "r_bottom_peaks.csv")


if __name__ == "__main__":
    main()
