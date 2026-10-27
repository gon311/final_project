"""E11-B 엣지 선명도 vs Ra (PREREG 4절, 탐색). 실행: /usr/bin/python3 e11b_edge_sharpness.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from paths import PROFILES, RESULTS, FIGURES, CACHE, UM_PER_PX
from e11a_edge_texture import spearman_perm

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SPECS = [f"NO{k}" for k in range(2, 13)]
UG = np.round(np.arange(-30, 30.0001, 0.1), 1)
K = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2); K /= K.sum()  # cv2.GaussianBlur((1,7), σ1.2)와 같은 커널


def gauss7(p):
    return np.convolve(np.pad(p, 3, mode="reflect"), K, mode="valid")


def esf_rows(P, L, flip, smooth):
    """프로파일 행들 -> (n, len(UG)) 정규화 ESF(물체 0, 배경 1, u<0이 배경쪽). 레벨 부족 열은 건너뛴다."""
    out, dropped = [], 0
    i = np.arange(P.shape[1], dtype=float) - 44
    for p, lab in zip(P, L):
        u = i - lab
        if flip:
            u, p = -u[::-1], p[::-1]
        if smooth:
            p = gauss7(p)
        bgm, obm = (u <= -25) & (u >= -40), (u >= 25) & (u <= 40)
        if bgm.sum() < 3 or obm.sum() < 3:
            dropped += 1; out.append(np.full(len(UG), np.nan)); continue
        bg, ob = np.median(p[bgm]), np.median(p[obm])
        out.append(np.interp(UG, u, (p - ob) / (bg - ob), left=np.nan, right=np.nan))
    return np.array(out), dropped


def width(esf):
    """0.5 교차점에서 바깥으로 0.9 / 0.1 교차점까지(선형 보간). 반환 (폭, 최대 기울기 |dESF/du|)."""
    ok = ~np.isnan(esf)
    if not ok.all():
        esf = np.interp(UG, UG[ok], esf[ok])
    c = int(np.flatnonzero((esf[:-1] >= 0.5) & (esf[1:] < 0.5))[0])  # 첫 0.5 하강 교차
    j = c
    while j > 0 and esf[j] < 0.9: j -= 1
    k = c
    while k < len(esf) - 2 and esf[k + 1] > 0.1: k += 1
    u90 = UG[j] + (esf[j] - 0.9) / (esf[j] - esf[j + 1]) * 0.1 if esf[j] != esf[j + 1] else UG[j]
    u10 = UG[k] + (esf[k] - 0.1) / (esf[k] - esf[k + 1]) * 0.1 if esf[k] != esf[k + 1] else UG[k]
    return float(u10 - u90), float(np.max(-np.gradient(esf, UG)))


if __name__ == "__main__":
    z = np.load(PROFILES, allow_pickle=True)
    fn, sp = z["filename"].astype(str), z["specimen"].astype(str)
    edges = pd.read_csv(CACHE / "edges_long.csv.gz", usecols=["filename", "단계", "시각"]).drop_duplicates("filename")
    stage = dict(zip(edges.filename, edges["단계"]))
    t = dict(zip(edges.filename, pd.to_timedelta(edges["시각"]).dt.total_seconds()))
    isMain = np.array([stage[f] == "본시험(암막)" for f in fn])
    rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")

    rows, ESF = [], {}
    for smooth in (True, False):
        for edge, flip in (("top", False), ("bot", True)):
            P, L = z[f"{edge}_profiles"].astype(float), z[f"{edge}_label"]
            for s in SPECS:
                m = isMain & (sp == s)
                E, dropped = esf_rows(P[m], L[m], flip, smooth)
                good = ~np.isnan(E).all(1)
                w, g = width(np.nanmean(E[good], 0))
                per = []
                for f in np.unique(fn[m]):
                    sel = (fn[m] == f) & good
                    per.append(width(np.nanmean(E[sel], 0))[0])
                ESF[(smooth, edge, s)] = np.nanmean(E[good], 0)
                rows.append(dict(specimen=s, edge=edge, smoothed=smooth, n_cols=int(good.sum()), n_dropped=dropped,
                                 width_px=w, width_um=w * UM_PER_PX, max_slope_per_px=g,
                                 width_img_std_px=float(np.std(per, ddof=1)),
                                 first_time_s=min(t[f] for f in np.unique(fn[m])),
                                 Ra=rough.at[s, "Ra"], Rz=rough.at[s, "Rz"]))
    by = pd.DataFrame(rows)
    by["order_rank"] = by.groupby(["edge", "smoothed"]).first_time_s.rank()
    by.to_csv(RESULTS / "E11B_by_specimen.csv", index=False)

    summ = []
    for (smooth, edge), d in by.groupby(["smoothed", "edge"]):
        rho, p = spearman_perm(d.Ra.to_numpy(), d.width_px.to_numpy())
        rz, pz = spearman_perm(d.Rz.to_numpy(), d.width_px.to_numpy())
        ro = float(np.corrcoef(rankdata(d.width_px), rankdata(d.order_rank))[0, 1])
        ra_o = float(np.corrcoef(rankdata(d.Ra), rankdata(d.order_rank))[0, 1])
        conf = abs(ro) >= 0.5
        text = (f"{edge} 10–90% 폭과 Ra의 Spearman ρ = {rho:+.2f}(단측 순열 p = {p:.3f}, n = 11, 탐색). "
                f"폭과 촬영 순서의 ρ = {ro:+.2f}" + (" → 촬영 순서(배율 드리프트) 교락 가능." if conf else "."))
        summ.append(dict(smoothed=smooth, edge=edge, rho_Ra=rho, p_Ra_one_sided=p, rho_Rz=rz, p_Rz_one_sided=pz,
                         rho_width_order=ro, rho_Ra_order=ra_o, confound_flag=conf,
                         is_main=(smooth and edge == "top"), conclusion=text))
    sm = pd.DataFrame(summ); sm.to_csv(RESULTS / "E11B_summary.csv", index=False)

    # 그림
    tb = by[(by.smoothed) & (by.edge == "top")].set_index("specimen")
    order = tb.Ra.sort_values().index
    cm = plt.cm.viridis(np.linspace(0, 1, len(order)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, lim in zip(axs, (30, 8)):
        for c, s in zip(cm, order):
            e = ESF[(True, "top", s)]; m = (UG >= -lim) & (UG <= lim)
            ax.plot(UG[m], e[m], color=c, lw=1, label=f"{s} Ra={tb.at[s, 'Ra']:.2f}")
        ax.set_xlabel("엣지 기준 위치 u (px, <0이 배경)"); ax.set_ylabel("정규화 밝기 (물체 0 ~ 배경 1)")
    axs[1].set_title("확대(±8px)"); axs[0].legend(fontsize=6, ncol=2)
    fig.suptitle("윗엣지 ESF (Ra 순서로 색칠, 평활 적용)"); fig.tight_layout(); fig.savefig(FIGURES / "E11B_esf_overlay.png", dpi=110); plt.close(fig)

    m_ = sm[sm.is_main].iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(tb.Ra, tb.width_px, s=30)
    for s, r in tb.iterrows(): ax.annotate(s, (r.Ra, r.width_px), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Ra (µm)"); ax.set_ylabel("윗엣지 10–90% 폭 (px)")
    ax.set_title(f"ρ={m_.rho_Ra:+.2f}, 단측 p={m_.p_Ra_one_sided:.3f}; 순서와 ρ={m_.rho_width_order:+.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "E11B_width_vs_Ra.png", dpi=110); plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(by[(by.smoothed) & (by.edge == "top")].drop(columns=["smoothed", "edge", "first_time_s"]).round(3).to_string(index=False))
    print(sm.drop(columns="conclusion").round(3).to_string(index=False)); print(sm[sm.is_main].conclusion.iloc[0])
    print(by[(by.edge == "bot") & by.smoothed][["specimen", "n_cols", "n_dropped", "width_px"]].round(3).to_string(index=False))
