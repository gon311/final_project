"""E4 그림 2장. (1) 정의별 셋업 중앙값 지름 — 묶음4(NO4)에서 grad_peak만 튀는 걸 보인다.
(2) 하단 문턱 sweep — 상단 고정, 하단 문턱 10~90%에서 셋업별 중앙값 지름이 얼마나
흔들리는지(팀원B 0.2b: "50% 문턱이 셋업마다 다른 단에 걸린다"와 같은 방향)."""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab.plotting import setup_korean_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    font_ok = setup_korean_font()
    label_kw = dict() if font_ok else dict()

    bench = pd.read_csv(os.path.join(HERE, "results", "edge_method_benchmark.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in bench.groupby("정의"):
        g = g.sort_values("묶음")
        style = "-" if name in ("cross50", "moment", "esf1") else "--"
        ax.plot(g["묶음"], g["diameter_med"], style, marker="o", ms=3, label=name)
    ax.set_xlabel("묶음" if font_ok else "setup (bundle)")
    ax.set_ylabel("지름 중앙값 (px)" if font_ok else "median diameter (px)")
    ax.set_title("정의별 셋업 중앙값 — 묶음4(NO4)에서 grad_peak만 튄다"
                  if font_ok else "median diameter by definition (NO4 outlier = grad_peak)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "definitions_by_setup.png"), dpi=140)
    plt.close(fig)

    sweep = pd.read_csv(os.path.join(HERE, "results", "bottom_threshold_sweep.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bundle, g in sweep.groupby("묶음"):
        g = g.sort_values("threshold_pct")
        ax.plot(g["threshold_pct"], g["diameter_med"], alpha=0.5, lw=1)
    med = sweep.groupby("threshold_pct")["diameter_med"].median()
    ax.plot(med.index, med.values, "k-", lw=2.5, label="13개 셋업 median" if font_ok else "median over 13 setups")
    ax.set_xlabel("하단 문턱 % (상단 50% 고정)" if font_ok else "bottom threshold % (top fixed at 50%)")
    ax.set_ylabel("지름 중앙값 (px)" if font_ok else "median diameter (px)")
    ax.set_title("하단 문턱 sweep — 셋업별 선, 13개 묶음"
                  if font_ok else "bottom threshold sweep (one line per setup)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "bottom_threshold_sweep.png"), dpi=140)
    plt.close(fig)
    print("저장: figures/definitions_by_setup.png, figures/bottom_threshold_sweep.png")


if __name__ == "__main__":
    main()
