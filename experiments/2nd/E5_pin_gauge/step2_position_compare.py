"""E5 2단계: 핀 측정 위치 세 방식을 비교한다. 추정 방식(강건 직선적합, `measure_section`과
동일한 로직)은 세 방식 모두 같게 두고, **측정에 쓰는 열 구간(창)만** 다르게 해서 창 선택의
효과만 비교한다. 세 방식 모두 `shaftlab.measure_pin()`을 130장에 직접 호출해서 얻는다
(캐시된 `pin_profile.csv`에서 다시 집계하지 않음 — Codex 2차 adversarial-review 반영:
런타임 API와 평가 산출물이 서로 다른 계산이 되는 걸 막는다).
  (a) 현 EDA 자동 구간: `position_method="auto"` ([xL+0.15·len, xR-0.15·len])
  (b) 이미지 고정 열: `position_method="fixed_cols"` ([9655, 9700])
  (c) 핀 상대 구간: `position_method="pin_relative"` ([xL+60, xL+105], `xR`로 클램프)
      — `shaftlab.measure_pin`의 기본값. 클램프돼 창이 좁아진 프레임은 `window_clipped`로 표시한다.
비교 지표: 묶음 안 sd, F1/F2/F3 초점구간 안 묶음간 범위, 묶음 1·2 대 3 이후 차이.

입력: 원본 이미지(`paths.IMAGE_DIR`), `paths.EDA_MEASURE_RAW_CSV`(분류표 메타데이터용)
산출물: results/pin_position_compare.csv, figures/pin_position_methods.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import measure_pin, paths  # noqa: E402
from shaftlab.plotting import setup_korean_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = {
    "method_a_auto": "auto",
    "method_b_fixed_cols": "fixed_cols",
    "method_c_pin_relative": "pin_relative",
}


def main():
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]  # eda_pin_gauge.py와 같은 기준, 0.3절(메모 사진 제외)
    files = sorted(meta.index)

    rows = []
    for fn in files:
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        row = dict(file=fn, 묶음=int(meta.loc[fn, "묶음"]), 초점=meta.loc[fn, "초점"])
        for col, pm in METHODS.items():
            r = measure_pin(gray, position_method=pm)
            row[col] = r["diameter_px"]
            if pm == "pin_relative":
                row["xL"] = r["xL"]
                row["xR"] = r["xR"]
                row["window_clipped"] = r["window_clipped"]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("file")
    out_path = os.path.join(HERE, "results", "pin_position_compare.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(out)}행)")

    clipped = out[out["window_clipped"]]
    print(f"\nwindow_clipped=True (pin_relative 창이 xR에 잘림) 이미지 {len(clipped)}장:")
    if len(clipped):
        print(clipped[["file", "묶음", "xL", "xR", "method_c_pin_relative"]].to_string(index=False))

    methods = list(METHODS.keys())
    bundle = out.groupby("묶음").agg(초점=("초점", "first"), **{f"{m}_mean": (m, "mean") for m in methods},
                                     **{f"{m}_sd": (m, "std") for m in methods},
                                     n_clipped=("window_clipped", "sum"))
    print("\n묶음별 평균/표준편차 (n_clipped = pin_relative 창이 잘린 프레임 수):\n", bundle.round(3).to_string())

    print("\n묶음 안 sd 중앙값(13묶음 median):")
    for m in methods:
        print(f"  {m}: {bundle[f'{m}_sd'].median():.3f}px")

    print("\nF1/F2/F3 초점구간 안 묶음간 범위(묶음 평균의 ptp):")
    for m in methods:
        spreads = {}
        for focus in ("F1", "F2", "F3"):
            vals = bundle.loc[bundle["초점"] == focus, f"{m}_mean"]
            spreads[focus] = float(vals.max() - vals.min()) if len(vals) > 1 else np.nan
        print(f"  {m}: F1={spreads['F1']:.3f} F2={spreads['F2']:.3f} F3={spreads['F3']:.3f}px")

    print("\n묶음 1·2 평균 대 묶음 3 이후 평균 차이:")
    for m in methods:
        g12 = bundle.loc[[1, 2], f"{m}_mean"].mean()
        g3plus = bundle.loc[bundle.index >= 3, f"{m}_mean"].mean()
        print(f"  {m}: 묶음1·2={g12:.3f}px, 묶음3+={g3plus:.3f}px, 차이={g12 - g3plus:+.3f}px")

    found_font = setup_korean_font()
    if found_font:
        labels = {"method_a_auto": "(a) 자동 구간", "method_b_fixed_cols": "(b) 고정 열 9655-9700",
                  "method_c_pin_relative": "(c) 핀 상대 s=60-105 (선정)"}
        xlabel, ylabel = "묶음", "핀 지름 (px)"
        title = "핀 측정 위치 방식 비교 (묶음별 평균 ± sd)"
        shift_note = "카메라 이동\n(묶음1-2 -> 3+)"
    else:
        labels = {"method_a_auto": "(a) auto", "method_b_fixed_cols": "(b) fixed cols 9655-9700",
                  "method_c_pin_relative": "(c) pin-relative s=60-105 (chosen)"}
        xlabel, ylabel = "bundle", "pin diameter (px)"
        title = "Pin position method comparison (mean +- sd per bundle)"
        shift_note = "camera shift\n(bundle 1-2 -> 3+)"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in methods:
        ax.errorbar(bundle.index, bundle[f"{m}_mean"], yerr=bundle[f"{m}_sd"], marker="o",
                     label=labels[m], capsize=3)
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1)
    ax.text(2.5, ax.get_ylim()[1], shift_note, fontsize=7, ha="center", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "figures", "pin_position_methods.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\n저장: {fig_path}")


if __name__ == "__main__":
    main()
