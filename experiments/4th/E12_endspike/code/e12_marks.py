"""사람이 눈으로 볼 수 있게 표시한 그림 2장 (분석 아님, 기존 결과와 사진을 그대로 그린다).
 (1) figures/E12_spike_marks.png : 시편별 8265 스파이크 위치(윗엣지 확대 + 엣지 추적선)
 (2) figures/E12_bright_band_not_flat.png : '평평하다고 가정한' 밝은 띠가 실제로는 질감이 있고 시편이 달라도 같은 모양이라는 것"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import FIGURES, PROFILES as CACHE, UM_PER_PX
from e12_edges import main, load, first_cross

plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
XC, SPECS = 8265, [f"NO{i}" for i in range(2, 13)]
z = np.load(CACHE, allow_pickle=True)
cen = {f: int(c) for f, c in zip(z["filename"].astype(str), z["top_center"])}
rows = {s: [r for r in main.itertuples() if r.specimen == s] for s in SPECS}


def trace(g, c, xs):
    return np.array([(lambda e: np.nan if e is None else e + c - 60)(first_cross(g[c - 60:c + 61, x])) for x in xs])


# ---------- (1) 시편별 스파이크 위치 ----------
xs = np.arange(XC - 40, XC + 41)
fig, axs = plt.subplots(4, 3, figsize=(15, 12.5)); axs = axs.ravel()
for ax, s in zip(axs, SPECS):
    R = rows[s]; T = []
    for k, r in enumerate(R):
        g = load(r); c = cen[r.filename]; e = trace(g, c, xs); T.append(e - np.nanmedian(e))
        if k == 0:
            e0 = e; yc = int(round(np.nanmedian(e))); crop = g[yc - 15:yc + 16, XC - 40:XC + 41]
    med = np.nanmedian(T, 0)
    ref = pd.Series(med).rolling(15, center=True, min_periods=5).median().values       # 스파이크를 뺀 기준선
    dev = med - ref; sp = float(np.nanmean(dev[(xs >= 8265) & (xs <= 8266)])) * UM_PER_PX
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255, extent=(xs[0] - .5, xs[-1] + .5, yc + 15.5, yc - 15.5), interpolation="nearest")
    ax.plot(xs, np.nanmedian(e0) + ref, "c--", lw=1.2, label="기준선(스파이크 제외)")
    ax.plot(xs, np.nanmedian(e0) + med, color="yellow", lw=1.5, label="엣지 위치(10장 중앙값)")
    ax.axvspan(8264.5, 8266.5, color="red", alpha=.28)
    if abs(sp) >= 8:
        ax.annotate(f"스파이크\n{sp:+.0f} um", xy=(8265.5, yc + (np.nanmedian(e0) - yc) + med[40] + 0), xytext=(8286, yc - 11), color="red", fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5), va="center")
    else:
        ax.text(8247, yc - 12, f"뚜렷한 스파이크 없음 ({sp:+.0f} um)", color="orange", fontsize=10, bbox=dict(fc="black", alpha=.7, ec="none"))
    ax.set_title(f"{s}  (윗엣지, x=8225~8305)", fontsize=12); ax.set_xlabel("x (px)"); ax.set_yticks([])
axs[0].legend(loc="lower left", fontsize=8)
for ax in axs[len(SPECS):]: ax.axis("off")
axs[-1].text(0.02, 0.62, "빨간 띠 = x 8265~8266\n노란 선 = 엣지 위치를 열마다 추적\n하늘색 점선 = 스파이크를 뺀 기준선\n(1px = 26 um, 스파이크는 1~2px)", transform=axs[-1].transAxes, fontsize=12, va="top")
fig.suptitle("시편별 8265 스파이크 위치 (원본 사진 확대 · 엣지 추적선)", fontsize=15); fig.tight_layout(rect=(0, 0, 1, .97))
fig.savefig(FIGURES / "E12_spike_marks.png", dpi=95); plt.close(fig)

# ---------- (2) 밝은 띠는 평평하지 않다 (절대 y 좌표: 같은 사진 위치를 비교) ----------
sel = ["NO2", "NO7", "NO12"]; X0, X1, YA, YB, YE = 8100, 8430, 3098, 3468, 3428     # 절대 y: 띠 = 3128~3308 (= 엣지중심 3428 −300 ~ −120)
crops = {s: load(rows[s][0])[YA:YB, X0:X1].astype(float) for s in sel}; times = {s: rows[s][0].시각 for s in sel}
band = {s: crops[s][3128 - YA:3308 - YA] for s in sel}; colm = {s: band[s].mean(0) for s in sel}
fig = plt.figure(figsize=(15, 11.5)); gs = fig.add_gridspec(3, 3, height_ratios=[3, 1.3, 1.5], hspace=.5, wspace=.12)
for j, s in enumerate(sel):
    ax = fig.add_subplot(gs[0, j]); ax.imshow(crops[s], cmap="gray", vmin=130, vmax=255, extent=(X0, X1, YB, YA), aspect="auto", interpolation="nearest")
    ax.add_patch(plt.Rectangle((X0, 3128), X1 - X0, 180, fill=False, ec="lime", lw=2.5)); ax.axvline(XC, color="red", lw=1.2, ls=":")
    ax.text(X0 + 4, 3122, "초록 사각형: '평평한 밝은 띠'로 가정했던 영역", color="lime", fontsize=10, va="bottom", fontweight="bold")
    ax.text(XC + 3, 3380, "← x=8265", color="red", fontsize=10); ax.text(X0 + 4, 3452, "시편(검은 부분)", color="white", fontsize=9)
    ax.set_title(f"{s} 사진 ({times[s]} 촬영) · 같은 사진 좌표", fontsize=11); ax.set_ylabel("사진 y (px)" if j == 0 else ""); ax.set_xlabel("x (px)")
corr = lambda p, q: np.corrcoef((p - p.mean(1, keepdims=True)).ravel(), (q - q.mean(1, keepdims=True)).ravel())[0, 1]
for j, s in enumerate(sel):
    ax = fig.add_subplot(gs[1, j]); ax.plot(np.arange(X0, X1), colm[s], lw=.8); ax.axvline(XC, color="red", ls=":")
    ax.set_title(f"띠 안 열별 평균 밝기 ({s}): 표준편차 {colm[s].std():.1f} gray", fontsize=10); ax.set_xlabel("x (px)"); ax.set_ylabel("gray")
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); b0 = band["NO7"]
ax.text(0.0, 0.97,
        f"• 초록 사각형 안은 '평평한 배경'이 아니라 미세한 섬유·종이 같은 질감입니다. 한 사진 안 밝기 표준편차 약 {b0.std():.0f} gray (평균 약 {b0.mean():.0f}).\n"
        f"• 시편이 달라도 같은 사진 위치의 질감은 같습니다: 픽셀 상관 NO2~NO7 {corr(band['NO2'], band['NO7']):.2f}, NO7~NO12 {corr(band['NO7'], band['NO12']):.2f}, NO2~NO12 {corr(band['NO2'], band['NO12']):.2f}\n"
        f"   (같은 시편 안 두 장의 상관이 0.80이라 0.80이 상한. 11개 시편 쌍 전체 중앙값 0.76, 최소 0.53). 질감은 시편이 아니라 사진(역광판 쪽)에 붙어 있습니다.\n"
        f"• (예비 관찰, 사후) 상관은 촬영 순서가 멀수록 낮아집니다(NO2~NO12 0.53): 질감이 사진 안에서 서서히 오른쪽으로 움직여 NO2→NO12(약 19분)에 약 1.7px 이동합니다(y는 거의 0).\n"
        f"   센서·JPEG에 붙은 고정 패턴이라면 움직이지 않을 것이므로 장면(역광판) 쪽 특징으로 보입니다.",
        transform=ax.transAxes, fontsize=12, va="top", linespacing=1.7)
fig.suptitle("밝은 띠는 평평하지 않다: 모든 시편 사진에 같은 질감이 있다", fontsize=15)
fig.savefig(FIGURES / "E12_bright_band_not_flat.png", dpi=95, bbox_inches="tight"); plt.close(fig)
print("saved; band corr NO2~NO7 %.2f NO7~NO12 %.2f NO2~NO12 %.2f" % (corr(band["NO2"], band["NO7"]), corr(band["NO7"], band["NO12"]), corr(band["NO2"], band["NO12"])))
