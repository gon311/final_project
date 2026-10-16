"""E7 그림: (1) clean MAE vs 완료 기준선(0.05px), 모델/엣지/변형별. (2) 증강 강도별
강건성(강도 축, mae_px). (3) 셋업 내 sigma 비교(teacher vs 모델)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab.plotting import setup_korean_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
CRITERION_PX = 0.05


def fig_clean_mae():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "cv_metrics.csv"))
    clean = df[(df["aug"] == "clean") & (~df["fold"].astype(str).str.startswith("ng_eval_only"))]
    g = clean.groupby(["edge", "head", "variant"])["mae_px"].mean().reset_index()
    g["label"] = g["edge"] + "/" + g["head"] + "/" + g["variant"]
    g = g.sort_values("mae_px")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4477aa" if h != "baseline_zero" else "#bbbbbb" for h in g["head"]]
    ax.barh(g["label"], g["mae_px"], color=colors)
    ax.axvline(CRITERION_PX, color="red", linestyle="--", linewidth=1, label=f"완료 기준 {CRITERION_PX}px")
    ax.set_xscale("log")  # baseline(수 px)과 모델(수십 분의 1px) 차이가 100배 이상이라 로그축
    ax.set_xlabel("teacher 대비 median |오차| (px, LOSO fold 평균, 로그축)")
    ax.set_title("E7: clean teacher 오차 — 모델/엣지/변형별")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "clean_mae_by_combo.png"), dpi=150)
    plt.close(fig)


def fig_robustness():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "cv_metrics.csv"))
    df = df[(df["head"] != "baseline_zero") & (~df["fold"].astype(str).str.startswith("ng_eval_only"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, edge in zip(axes, ("top", "bottom")):
        sub = df[(df["edge"] == edge) & (df["variant"] == "raw")]
        g = sub.groupby(["head", "aug", "intensity"], dropna=False)["mae_px"].mean().reset_index()
        g["x"] = np.where(
            g["intensity"].isna(), g["aug"].astype(str),
            g["aug"].astype(str) + "=" + g["intensity"].astype(str),
        )
        for head, gg in g.groupby("head"):
            ax.plot(gg["x"], gg["mae_px"], marker="o", label=head)
        ax.axhline(CRITERION_PX, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{edge} (variant=raw)")
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
    axes[0].set_ylabel("median |오차| (px)")
    fig.suptitle("E7: 증강 강도별 강건성 (clean 포함 안 함)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "robustness_by_aug.png"), dpi=150)
    plt.close(fig)


def fig_sigma():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "setup_sigma.csv"))
    g = df.groupby("model")["sigma_px"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2a9d8f" if m.startswith("teacher") else "#4477aa" for m in g.index]
    ax.barh(g.index, g.values, color=colors)
    ax.set_xlabel("셋업 내 10장 사이 지름 σ (px, 평균)")
    ax.set_title("E7: 셋업 내 σ — teacher vs 모델 재계산")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "setup_sigma.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_korean_font()  # axes.unicode_minus=False도 같이 설정한다(plotting.py)
    fig_clean_mae()
    fig_robustness()
    fig_sigma()
    print("저장:", FIG_DIR)


if __name__ == "__main__":
    main()
