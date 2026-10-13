"""E5 1단계: 열별 핀 지름 프로파일(`pin_profile.csv`)에서 팀원B 결과(지름이 오른쪽으로
140px 가는 동안 +2.6px, 위 경계만 100px당 -1.3~-1.7px 기울고 아래는 평평함, 1011_pincol)가
재현되는지 확인한다.

입력: results/pin_profile.csv (scan_pins.py 산출물)
산출물: results/pin_slope_by_bundle.csv, figures/pin_width_profile.png
"""
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

# 팀원B 1011_pincol/_out/analyze.txt 수치 (인용만, 재계산하지 않음)
TEAMB_TOP_SLOPE_RANGE = (-1.72, -1.30)  # px/100px, 13묶음 범위
TEAMB_BOT_SLOPE_RANGE = (-0.14, 0.24)
TEAMB_S_PROFILE = {  # s 구간 시작 -> 130장 평균 폭(자기 중앙값 뺀 값)
    20: -0.62, 40: -0.40, 60: -0.23, 80: -0.07, 100: 0.09, 120: 0.37, 140: 1.27, 160: 1.96,
}


def slope_per_100px(s, y):
    if len(s) < 3:
        return np.nan
    return float(np.polyfit(s, y, 1)[0]) * 100


def main():
    prof = pd.read_csv(os.path.join(HERE, "results", "pin_profile.csv"), encoding="utf-8-sig")

    rows = []
    for fn, g in prof.groupby("file"):
        g = g.sort_values("s")
        pin_len = g["s"].max()
        m = (g["s"] >= 20) & (g["s"] <= pin_len)
        rows.append(dict(
            file=fn, 묶음=int(g["묶음"].iloc[0]), 초점=g["초점"].iloc[0], pin_len=int(pin_len),
            top_slope_cross=slope_per_100px(g.loc[m, "s"], g.loc[m, "top_cross"]),
            bot_slope_cross=slope_per_100px(g.loc[m, "s"], g.loc[m, "bot_cross"]),
            top_slope_peak=slope_per_100px(g.loc[m, "s"], g.loc[m, "top_peak"]),
            bot_slope_peak=slope_per_100px(g.loc[m, "s"], g.loc[m, "bot_peak"]),
        ))
    per_image = pd.DataFrame(rows)

    bundle = per_image.groupby("묶음").agg(
        초점=("초점", "first"), pin_len_med=("pin_len", "median"),
        top_slope_cross=("top_slope_cross", "median"), bot_slope_cross=("bot_slope_cross", "median"),
        top_slope_peak=("top_slope_peak", "median"), bot_slope_peak=("bot_slope_peak", "median"),
    )
    out_path = os.path.join(HERE, "results", "pin_slope_by_bundle.csv")
    bundle.to_csv(out_path, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(bundle.round(3).to_string())

    print(f"\n위 경계 기울기(cross50) 범위: {bundle['top_slope_cross'].min():.2f} ~ {bundle['top_slope_cross'].max():.2f} px/100px"
          f" (팀원B: {TEAMB_TOP_SLOPE_RANGE[0]} ~ {TEAMB_TOP_SLOPE_RANGE[1]})")
    print(f"아래 경계 기울기(cross50) 범위: {bundle['bot_slope_cross'].min():.2f} ~ {bundle['bot_slope_cross'].max():.2f} px/100px"
          f" (팀원B: {TEAMB_BOT_SLOPE_RANGE[0]} ~ {TEAMB_BOT_SLOPE_RANGE[1]})")

    # s-구간별 평균 프로파일 (각 이미지 자기 중앙값을 뺀 값, 팀원B와 같은 정의)
    def s_profile(width_col):
        out = {}
        for fn, g in prof.groupby("file"):
            g = g.sort_values("s")
            pin_len = g["s"].max()
            m_full = (g["s"] >= 0) & (g["s"] <= pin_len)
            ref = g.loc[m_full & (g["s"] >= 20), width_col].median()
            for s, w in zip(g.loc[m_full, "s"], g.loc[m_full, width_col]):
                b = int(s // 20 * 20)
                out.setdefault(b, []).append(w - ref)
        return {b: (np.mean(v), len(v)) for b, v in out.items() if len(v) > 30}

    cross_profile = s_profile("width_cross")
    peak_profile = s_profile("width_peak")

    print("\ns-구간별 평균 폭(자기 중앙값 뺀 값, cross50) vs 팀원B:")
    for s0 in sorted(cross_profile):
        teamb = TEAMB_S_PROFILE.get(s0 if s0 != 0 else 20)
        mean, n = cross_profile[s0]
        teamb_str = f"{teamb:+.2f}" if teamb is not None else "  -  "
        print(f"  s {s0:3d}~{s0 + 19:3d}: {mean:+.2f}px (n={n})  | 팀원B {teamb_str}")

    found_font = setup_korean_font()
    if found_font:
        label_cross, label_peak, label_teamb = "cross50 (우리)", "grad_peak (우리)", "cross50 (팀원B, 인용)"
        label_window = "선정된 측정 구간 s=60-105"
        xlabel, ylabel = "s = x - xL (px)", "폭 - 이미지 자기 중앙값(s≥20) (px)"
        title = "핀 폭 프로파일: V블록 쪽으로 갈수록 위 경계가 올라간다"
    else:
        label_cross, label_peak, label_teamb = "cross50 (ours)", "grad_peak (ours)", "cross50 (teamB, cited)"
        label_window = "selected window s=60-105"
        xlabel, ylabel = "s = x - xL (px)", "width - per-image median (s>=20) (px)"
        title = "Pin width profile: top edge rises toward the V-block side"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for profile, label, color in ((cross_profile, label_cross, "tab:blue"), (peak_profile, label_peak, "tab:orange")):
        xs = sorted(profile)
        ys = [profile[x][0] for x in xs]
        ax.plot([x + 10 for x in xs], ys, marker="o", color=color, label=label)
    teamb_xs = sorted(TEAMB_S_PROFILE)
    teamb_ys = [TEAMB_S_PROFILE[x] for x in teamb_xs]
    ax.plot([x + 10 for x in teamb_xs], teamb_ys, marker="x", linestyle="--", color="gray", label=label_teamb)
    ax.axvspan(60, 105, color="tab:green", alpha=0.12, label=label_window)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "figures", "pin_width_profile.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\n저장: {fig_path}")


if __name__ == "__main__":
    main()
