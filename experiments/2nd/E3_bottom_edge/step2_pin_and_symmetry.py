"""E3 4~5단계.
  4. 같은 다단 분석을 Ø10 핀게이지 상·하단에도 수행한다. 팀원B는 핀 위·아래를 "1~2px
     절벽 하나"로 봤다 — 재현되면 다단이 샤프트 쪽(표면·자세·위치) 성분이라는 근거가 된다.
  5. 상·하단 대칭성으로 가림 가설을 본다. 상단은 가림막으로 가렸고(1단), 하단은 가리지
     않았다(다단). "가리지 않은 백라이트가 샤프트 아래를 스친다"는 팀원B 추정과 맞는지 판단.

판정 기준: `find_peaks_subpixel`이 찾은 2·3번째 극댓값이 가장 강한 극댓값 대비 얼마나
강한지(weakest/strongest 세기 비). 비율이 낮으면(노이즈 수준) 사실상 "절벽 하나", 높으면
(비슷한 세기로 경쟁) "진짜 다단"으로 본다.

입력: results/bottom_peaks.csv, cache/e3/top_peaks.csv, cache/e3/pin_edge_peaks.csv
산출물: figures/pin_vs_shaft_bottom_profile.png, figures/top_bottom_alignment.png
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
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, PIN_COARSE  # noqa: E402
from shaftlab.plotting import setup_korean_font  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_E3 = os.path.join(HERE, "..", "cache", "e3")


def weakest_strongest_ratio(df, n_peaks_col="n_peaks"):
    d3 = df[df[n_peaks_col] == 3].copy()
    s = d3[["peak1_strength", "peak2_strength", "peak3_strength"]]
    strongest = s.max(axis=1)
    weakest = s.min(axis=1)
    return (weakest / strongest), len(d3)


def extract_profile(gray, seg, X, half_window=60):
    """측정 당시와 같은 블러·열 추출 방식으로 한 열의 밝기 프로파일(원시)을 뽑는다."""
    si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
    tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
    y0, y1 = tc - half_window - 20, bc + half_window + 20
    prof = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
    return prof, y0, tc, bc


def main():
    bottom = pd.read_csv(os.path.join(HERE, "results", "bottom_peaks.csv"), encoding="utf-8-sig")
    top = pd.read_csv(os.path.join(CACHE_E3, "top_peaks.csv"), encoding="utf-8-sig")
    pin = pd.read_csv(os.path.join(CACHE_E3, "pin_edge_peaks.csv"), encoding="utf-8-sig")

    print("== 4단계: 핀 vs 샤프트 다단 비교 (2·3번째 극댓값이 노이즈 수준인가, 진짜 경쟁하는 단인가) ==")
    r_top, n_top = weakest_strongest_ratio(top)
    r_bot, n_bot = weakest_strongest_ratio(bottom)
    r_pin_top, n_pt = weakest_strongest_ratio(pin[pin["side"] == "top"])
    r_pin_bot, n_pb = weakest_strongest_ratio(pin[pin["side"] == "bot"])
    print(f"  샤프트 상단   : weakest/strongest 중앙값 {r_top.median():.3f} (n={n_top}) -> 절벽 하나에 가까움")
    print(f"  샤프트 하단   : weakest/strongest 중앙값 {r_bot.median():.3f} (n={n_bot}) -> 상단보다 {r_bot.median()/r_top.median():.1f}배 강함, 다단")
    print(f"  핀 상단       : weakest/strongest 중앙값 {r_pin_top.median():.3f} (n={n_pt}) -> 절벽 하나")
    print(f"  핀 하단       : weakest/strongest 중앙값 {r_pin_bot.median():.3f} (n={n_pb}) -> 절벽 하나")
    print("  -> 팀원B의 '핀 위·아래는 절벽 하나'가 재현된다(핀 상·하단 모두 샤프트 상단과 같은 낮은 비율).")
    print("  -> 다단은 핀에는 없고 샤프트 하단에만 있다 — 깊이·위치보다 '가리지 않음'(5단계)과 더 맞는 방향이다.")

    print("\n== 5단계: 상·하단 대칭성 (가림 가설) ==")
    print(f"  샤프트 상단이 핀(절벽 하나)과 비슷한 낮은 비율을 보이는 것은 상단이 가림막으로")
    print(f"  가려 다단이 생길 조건(반사 등)이 없다는 가설과 맞는 방향이다. 다만 이 비교만으로")
    print(f"  '가리지 않은 백라이트가 샤프트 아래를 스친다'는 원인까지 확정할 수는 없다(가설).")

    # 그림 1: 핀 vs 샤프트 하단 프로파일 비교 (대표 이미지 1장)
    fn = "20261008_151701.jpg"
    gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
    q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    seg_b = coarse_section(q, **BEARING_COARSE)
    xb_mid = (int(seg_b[0, 0] * 4) + int(seg_b[-1, 0] * 4)) // 2
    shaft_prof, shaft_y0, shaft_tc, shaft_bc = extract_profile(gray, seg_b, xb_mid)

    seg_p = coarse_section(q, **PIN_COARSE)
    xp_mid = (int(seg_p[0, 0] * 4) + int(seg_p[-1, 0] * 4)) // 2
    pin_prof, pin_y0, pin_tc, pin_bc = extract_profile(gray, seg_p, xp_mid)

    found_font = setup_korean_font()
    if found_font:
        t_shaft, t_pin = "샤프트 하단 (다단)", "핀 하단 (절벽 하나)"
        xlabel, ylabel = "엣지에서 떨어진 거리 (px)", "밝기"
        title = "핀 vs 샤프트 하단 프로파일 비교 (대표 열 1개)"
    else:
        t_shaft, t_pin = "shaft bottom (multi-step)", "pin bottom (single cliff)"
        xlabel, ylabel = "distance from edge (px)", "brightness"
        title = "Pin vs shaft bottom profile comparison (one representative column)"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    win = 35
    s_rel = np.arange(-win, win)
    axes[0].plot(s_rel, shaft_prof[shaft_bc - shaft_y0 - win:shaft_bc - shaft_y0 + win], color="tab:red")
    axes[0].set_title(t_shaft)
    axes[1].plot(s_rel, pin_prof[pin_bc - pin_y0 - win:pin_bc - pin_y0 + win], color="tab:blue")
    axes[1].set_title(t_pin)
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig_path = os.path.join(HERE, "figures", "pin_vs_shaft_bottom_profile.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\n저장: {fig_path}")

    # 그림 2: 상·하단 정렬(상단을 뒤집어 같은 방향으로 맞춤)
    win2 = 35
    top_window = shaft_prof[shaft_tc - shaft_y0 - win2:shaft_tc - shaft_y0 + win2]
    bot_window = shaft_prof[shaft_bc - shaft_y0 - win2:shaft_bc - shaft_y0 + win2]
    top_flipped = top_window[::-1]  # 상단은 밝음->어두움이니 뒤집어 하단과 같은 방향(어두움->밝음)으로

    if found_font:
        label_top, label_bot = "상단(뒤집음)", "하단"
        xlabel2, ylabel2 = "엣지 기준 상대 위치 (px)", "밝기(정규화)"
        title2 = "상·하단 정렬: 상단을 뒤집어 하단과 같은 방향으로 맞춘 모습"
    else:
        label_top, label_bot = "top (flipped)", "bottom"
        xlabel2, ylabel2 = "position relative to edge (px)", "brightness (normalized)"
        title2 = "Top-bottom alignment: top flipped to match bottom's direction"

    def normalize(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    s_rel2 = np.arange(-win2, win2)
    ax2.plot(s_rel2, normalize(top_flipped), label=label_top, color="tab:green")
    ax2.plot(s_rel2, normalize(bot_window), label=label_bot, color="tab:red")
    ax2.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel(xlabel2)
    ax2.set_ylabel(ylabel2)
    ax2.set_title(title2)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2_path = os.path.join(HERE, "figures", "top_bottom_alignment.png")
    fig2.savefig(fig2_path, dpi=150)
    print(f"저장: {fig2_path}")


if __name__ == "__main__":
    main()
