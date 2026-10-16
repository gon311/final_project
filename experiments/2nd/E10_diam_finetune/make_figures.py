"""E10 그림 3장: `evaluate.py`/`step_diag.py` 산출물을 읽어서만 그린다(재계산 없음).
`shaftlab.plotting.setup_korean_font()`(E5·E6·E7과 같은 공용 헬퍼)로 한글 폰트를 찾고,
못 찾으면 영어 라벨로 대체한다(E6 `make_figures.py`와 같은 패턴)."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
sys.path.insert(0, os.path.join(D.REPO2ND))
from shaftlab.plotting import setup_korean_font  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def fig_mae_vs_baseline(font_ok: bool):
    summary = pd.read_csv(os.path.join(D.RESULTS_DIR, "summary.csv"), encoding="utf-8-sig")
    baseline = summary.loc[summary["variant"] == "0_평균찍기(baseline)", "MAE_um_model_direct"].iloc[0]
    rest = summary[summary["variant"] != "0_평균찍기(baseline)"].copy()
    rest = rest.sort_values("MAE_um_model_direct")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#6b7280" if v.startswith("TEACHER_") else "#2563eb" for v in rest["variant"]]
    ax.barh(rest["variant"], rest["MAE_um_model_direct"], color=colors)
    ax.axvline(baseline, color="#dc2626", linestyle="--", linewidth=1.5,
               label=(f"평균 찍기 baseline ({baseline:.1f}um)" if font_ok
                      else f"mean-guess baseline ({baseline:.1f}um)"))
    ax.set_xlabel("MAE (um), model_direct, LOSO")
    ax.set_title("변형별 MAE vs 평균 찍기 baseline (회색=teacher, 파란색=파인튜닝 변형)" if font_ok
                  else "MAE per variant vs mean-guess baseline (gray=teacher, blue=finetuned)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mae_vs_baseline.png"), dpi=150)
    plt.close(fig)


def fig_step_occupancy(font_ok: bool):
    occ = pd.read_csv(os.path.join(D.RESULTS_DIR, "step_occupancy.csv"), encoding="utf-8-sig")
    overall = occ.groupby(["source", "step"])["n"].sum().reset_index()
    overall["fraction"] = overall["n"] / overall.groupby("source")["n"].transform("sum")
    sources = ["teacher", "pretrain"] + sorted(s for s in overall["source"].unique()
                                                if s not in ("teacher", "pretrain"))
    steps = ["1", "2", "3", "ambiguous"]
    pivot = overall.pivot(index="source", columns="step", values="fraction").reindex(sources)
    pivot = pivot.reindex(columns=[c for c in steps if c in pivot.columns]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot))
    palette = {"1": "#2563eb", "2": "#f59e0b", "3": "#10b981", "ambiguous": "#9ca3af"}
    for step in pivot.columns:
        label = f"{'단' if font_ok else 'step'} {step}"
        ax.bar(pivot.index, pivot[step], bottom=bottom, label=label, color=palette.get(step, "#999999"))
        bottom += pivot[step].to_numpy()
    ax.set_ylabel("점유율" if font_ok else "fraction")
    ax.set_title("하단 단(step) 점유율 — teacher/사전학습/파인튜닝 변형별" if font_ok
                  else "Bottom-edge step occupancy — teacher / pretrain / finetuned variants")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step_occupancy.png"), dpi=150)
    plt.close(fig)


def fig_edge_shift(font_ok: bool):
    es = pd.read_csv(os.path.join(D.RESULTS_DIR, "edge_shift.csv"), encoding="utf-8-sig")
    es = es.sort_values("bot_shift_median_px")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(es))
    width = 0.35
    ax.bar(x - width / 2, es["top_shift_median_px"], width,
           label=("상단 median" if font_ok else "top median"), color="#2563eb")
    ax.bar(x + width / 2, es["bot_shift_median_px"], width,
           label=("하단 median" if font_ok else "bottom median"), color="#f59e0b")
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2,
               label=("해석 규칙 문턱(1px)" if font_ok else "interpretation threshold (1px)"))
    ax.set_xticks(x)
    ax.set_xticklabels(es["variant"], rotation=45, ha="right")
    ax.set_ylabel("teacher 대비 |이동|의 fold 중앙값(px)" if font_ok
                  else "fold-median |shift| vs teacher (px)")
    ax.set_title("변형별 열별 엣지 이동량(teacher 대비)" if font_ok
                  else "Per-column edge shift vs teacher, by variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "edge_shift.png"), dpi=150)
    plt.close(fig)


def main():
    global FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=None,
                     help="evaluate.py/step_diag.py 결과 CSV 위치(기본: E10_diam_finetune/results)")
    ap.add_argument("--figures-dir", type=str, default=None,
                     help="그림 저장 위치(기본: E10_diam_finetune/figures)")
    args = ap.parse_args()
    if args.results_dir:
        D.set_run_dirs(results_dir=args.results_dir)
    if args.figures_dir:
        FIG_DIR = args.figures_dir

    os.makedirs(FIG_DIR, exist_ok=True)
    font_ok = setup_korean_font() is not None
    fig_mae_vs_baseline(font_ok)
    fig_step_occupancy(font_ok)
    fig_edge_shift(font_ok)
    print(f"저장: {FIG_DIR}/mae_vs_baseline.png, step_occupancy.png, edge_shift.png (한글 폰트: {font_ok})")


if __name__ == "__main__":
    main()
