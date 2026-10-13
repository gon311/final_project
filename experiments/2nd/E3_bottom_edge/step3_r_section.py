"""E3 6단계: 우측 직선부(R_) 하단에서도 베어링부 하단과 같은 다단 구조가 나타나는지 확인한다.

입력: cache/e3/r_bottom_peaks.csv, results/bottom_peaks.csv
산출물: results/r_bottom_features_by_setup.csv
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_E3 = os.path.join(HERE, "..", "cache", "e3")


def weakest_strongest_ratio(df):
    d3 = df[df["n_peaks"] == 3].copy()
    s = d3[["peak1_strength", "peak2_strength", "peak3_strength"]]
    return (s.min(axis=1) / s.max(axis=1)), len(d3)


def main():
    r = pd.read_csv(os.path.join(CACHE_E3, "r_bottom_peaks.csv"), encoding="utf-8-sig")
    bottom = pd.read_csv(os.path.join(HERE, "results", "bottom_peaks.csv"), encoding="utf-8-sig")

    r_ratio, n_r = weakest_strongest_ratio(r)
    b_ratio, n_b = weakest_strongest_ratio(bottom)
    print(f"R_(우측 직선부) 하단: weakest/strongest 중앙값 {r_ratio.median():.3f} (n={n_r})")
    print(f"베어링부 하단        : weakest/strongest 중앙값 {b_ratio.median():.3f} (n={n_b})")
    ratio_of_ratios = r_ratio.median() / b_ratio.median()
    if ratio_of_ratios > 0.7:
        verdict = "비슷하다 -> 같은 다단 구조가 우측 직선부에도 나타난다(가설 지지)."
    else:
        verdict = "베어링부보다 약하다 -> 다단이 베어링부에 더 뚜렷하다."
    print(f"-> 비율의 비율 {ratio_of_ratios:.2f}: {verdict}")

    agg = r.groupby("묶음").agg(
        초점=("초점", "first"), n_cols=("x", "count"),
        frac_3steps=("n_peaks", lambda s: float((s == 3).mean())),
        edge_width_10_90_med=("edge_width_10_90", "median"),
        inner_ratio_med=("inner_ratio", "median"),
    )
    out_path = os.path.join(HERE, "results", "r_bottom_features_by_setup.csv")
    agg.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n저장: {out_path}\n")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
