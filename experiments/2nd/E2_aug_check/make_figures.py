"""E2 그림 2장. (1) 증강×강도별 |Δ상단|·|Δ하단|(cross50) 중앙값 막대 + 0.03px 판정선.
(2) 해상도 조건별 |Δ외경|(cross50) 분포."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab.plotting import setup_korean_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.03


def main():
    font_ok = setup_korean_font()
    crit = pd.read_csv(os.path.join(HERE, "results", "completion_criteria.csv"), encoding="utf-8-sig")
    photometric = crit[~crit["aug"].isin(["rotation_deg", "subpixel_shift_y_px"])].copy()
    photometric["label"] = photometric["aug"] + "=" + photometric["intensity"].astype(str)
    photometric = photometric.sort_values(["aug", "intensity"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(photometric))
    w = 0.35
    ax.bar(x - w / 2, photometric["top_cross50_med"], w, label="|Δ상단| median" if font_ok else "|d top| median")
    ax.bar(x + w / 2, photometric["bot_cross50_med"], w, label="|Δ하단| median" if font_ok else "|d bottom| median")
    ax.axhline(THRESH, color="red", ls="--", lw=1,
               label=f"판정선 {THRESH}px" if font_ok else f"threshold {THRESH}px")
    ax.set_xticks(x, photometric["label"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("|Δ| median (px, cross50)" if font_ok else "|delta| median (px)")
    ax.set_title("증강별 |Δ상단|·|Δ하단| 중앙값 (cross50)" if font_ok else "Median |delta| top/bottom by augmentation")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "aug_delta_by_config.png"), dpi=140)
    plt.close(fig)

    df = pd.read_csv(os.path.join(HERE, "results", "aug_effect.csv"), encoding="utf-8-sig")
    res = pd.read_csv(os.path.join(HERE, "results", "resolution_check.csv"), encoding="utf-8-sig")
    res = res[res["ok"]]
    res["label"] = res["condition"] + " " + res["phase"]
    order = ["2x2 (0, 0)", "2x2 (1, 0)", "2x2 (0, 1)", "2x2 (1, 1)", "4x4 (0, 0)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [res[res["label"] == o]["d_diam_cross50_px"].abs().to_numpy() for o in order]
    ax.boxplot(data, tick_labels=order)
    ax.axhline(THRESH, color="red", ls="--", lw=1, label=f"판정선 {THRESH}px" if font_ok else f"threshold {THRESH}px")
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("|Δ외경| (px, cross50)" if font_ok else "|delta diameter| (px)")
    n_imgs = res["file"].nunique()
    ax.set_title(f"해상도 조건별 |Δ외경| 분포 ({n_imgs}장)" if font_ok else f"Resolution condition |delta diameter| ({n_imgs} images)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "resolution_delta.png"), dpi=140)
    plt.close(fig)
    print("저장: figures/aug_delta_by_config.png, figures/resolution_delta.png")


if __name__ == "__main__":
    main()
