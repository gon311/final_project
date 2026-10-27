"""E11-A 시편 고유 엣지 굴곡 vs Rz (PREREG 3절). 실행: /usr/bin/python3 e11a_edge_texture.py"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from paths import CACHE, RESULTS, FIGURES, SEED, UM_PER_PX

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SPECS = [f"NO{k}" for k in range(2, 13)]
COMBOS = [c for c in itertools.combinations(range(10), 5) if 0 in c]  # 126가지, A쪽이 0번 이미지 포함
assert len(COMBOS) == 126
SPLITS_B = [tuple(i for i in range(10) if i not in c) for c in COMBOS]
rng0 = np.random.default_rng(SEED)
CROSS_IDX = np.sort(rng0.choice(126, 50, replace=False))  # 모든 시편 쌍이 같은 50가지 나눔을 쓴다
MIN_CROSS_COLS = 30

edges = pd.read_csv(CACHE / "edges_long.csv.gz")
rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")
main = edges[edges["단계"] == "본시험(암막)"]
remount = edges[edges["단계"] != "본시험(암막)"]


def detrend(x, y, deg):
    """강건 추세 제거: 1차 적합 -> |r|>=4*1.4826*MAD 제외 -> deg차 재적합. 잔차는 전 열에 대해 최종 적합 기준."""
    u = (x - 8100.0) / 100.0
    r = y - np.polyval(np.polyfit(u, y, 1), u)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    k = np.abs(r) < 4 * 1.4826 * mad
    return y - np.polyval(np.polyfit(u[k], y[k], deg), u)


def build(spec, col, deg):
    """시편 spec의 본시험 10장 잔차 행렬 (10, n_grid), 격자 x, 보간 여부."""
    imgs = [g for _, g in main[main.specimen == spec].sort_values(["시각", "filename"]).groupby("filename", sort=False)]
    imgs.sort(key=lambda g: g["filename"].iloc[0])
    assert len(imgs) == 10
    res = [(g["x"].to_numpy(), detrend(g["x"].to_numpy(float), g[col].to_numpy(), deg)) for g in imgs]
    inter = sorted(set.intersection(*[set(x) for x, _ in res]))
    n_orig = int(np.median([len(x) for x, _ in res]))
    if len(inter) >= 0.8 * n_orig:
        grid, interp = np.array(inter), False
        R = np.stack([r[np.isin(x, grid)] for x, r in res])
    else:  # PREREG 3.2: 80% 미만이면 2px 격자로 보간 (격자 = 이미지별 x범위 교집합, 가장 큰 시작점부터 2px)
        lo, hi = max(x.min() for x, _ in res), min(x.max() for x, _ in res)
        grid, interp = np.arange(lo, hi + 1, 2), True
        R = np.stack([np.interp(grid, x, r) for x, r in res])
    return R, grid, interp


def cov(a, b):
    return float(np.cov(a, b, ddof=1)[0, 1])


def analyze(col, deg):
    D = {s: build(s, col, deg) for s in SPECS}
    # split-half
    W = {}
    for s, (R, grid, _) in D.items():
        S, r = [], []
        for ca, cb in zip(COMBOS, SPLITS_B):
            A, B = R[list(ca)].mean(0), R[list(cb)].mean(0)
            S.append(cov(A, B)); r.append(np.corrcoef(A, B)[0, 1])
        W[s] = (np.array(S), np.array(r))
    # 고정 패턴 대조: 같은 절대 x의 교집합 열, 순서쌍 (p, q): p의 A쪽 5장 평균 vs q의 B쪽 5장 평균
    pairs, Sc = [], {s: [] for s in SPECS}
    for p, q in itertools.permutations(SPECS, 2):
        Rp, gp, _ = D[p]; Rq, gq, _ = D[q]
        common = np.intersect1d(gp, gq)
        if len(common) < MIN_CROSS_COLS:
            pairs.append(dict(p=p, q=q, n_cols=len(common), S_cross=np.nan, r_cross=np.nan, used=False)); continue
        ip, iq = np.isin(gp, common), np.isin(gq, common)
        sc, rc = [], []
        for k in CROSS_IDX:
            a = Rp[list(COMBOS[k])][:, ip].mean(0); b = Rq[list(SPLITS_B[k])][:, iq].mean(0)
            sc.append(cov(a, b)); rc.append(np.corrcoef(a, b)[0, 1])
        pairs.append(dict(p=p, q=q, n_cols=len(common), S_cross=np.median(sc), r_cross=np.median(rc), used=True))
        Sc[p].append((np.median(sc), np.median(rc)))
    rows = []
    for s in SPECS:
        S, r = W[s]
        cs = [v[0] for v in Sc[s]]; cr = [v[1] for v in Sc[s]]
        S_cross = float(np.median(cs)) if cs else np.nan
        S_spec = float(np.median(S)) - S_cross
        a_px = float(np.sqrt(max(S_spec, 0)))
        rm = float(np.median(r))
        rows.append(dict(specimen=s, edge=col[2:], detrend=f"poly{deg}", n_cols=len(D[s][1]), interpolated=D[s][2],
                         r_AB_med=rm, SB_rel=2 * rm / (1 + rm), S_within_med=float(np.median(S)),
                         S_cross_med=S_cross, n_cross_pairs=len(cs), r_cross_med=float(np.median(cr)) if cr else np.nan,
                         S_spec=S_spec, a_spec_px=a_px, a_spec_um=a_px * UM_PER_PX,
                         p05=float(np.percentile(S - S_cross, 5)), Rz=rough.at[s, "Rz"], Ra=rough.at[s, "Ra"],
                         Rz_Ra=rough.at[s, "Rz_Ra"]))
    df = pd.DataFrame(rows)
    df["has_specific"] = df["p05"] > 0
    return df, pd.DataFrame(pairs).assign(edge=col[2:], detrend=f"poly{deg}"), D


def spearman_perm(x, y, n=10000, seed=SEED):
    """Spearman rho와 단측(양) 순열 p = (count+1)/(n+1)."""
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rg = np.random.default_rng(seed)
    perm = np.stack([rg.permutation(ry) for _ in range(n)])
    rp = np.array([np.corrcoef(rx, row)[0, 1] for row in perm])
    return rho, float((np.sum(rp >= rho - 1e-12) + 1) / (n + 1))


DESC = {
    1: "사진 윗엣지의 시편 고유 굴곡이 Rz 순서를 따른다. 거칠기의 장파장 성분이 사진에 보일 가능성(n = 11, 탐색).",
    2: "시편 고유 굴곡은 있으나 Rz와의 관계는 이 데이터로 판단 불가(파상도·진원도·이물 등 다른 원인 가능).",
    3: "이 셋업 해상도에서 표면 굴곡은 측정 잡음·고정 패턴 아래에 있다.",
}
PREMISE = "(전제: 이미지는 약 0.1mm보다 긴 파장의 굴곡만 보며, 거칠기는 원주 한 위치의 1회 측정이다.)"

if __name__ == "__main__":
    all_df, all_pairs, summ, Dmain = [], [], [], None
    for col in ("y_top", "y_bot"):
        for deg in (2, 1):
            df, pr, D = analyze(col, deg)
            all_df.append(df); all_pairs.append(pr)
            n_has = int(df.has_specific.sum())
            for var in ("Rz", "Ra", "Rz_Ra"):
                rho, p = spearman_perm(df[var].to_numpy(), df["a_spec_px"].to_numpy())
                summ.append(dict(edge=col[2:], detrend=f"poly{deg}", rough_var=var, n_spec=len(df), n_has_specific=n_has,
                                 stage1=n_has >= 6, rho=rho, p_one_sided=p,
                                 is_main=(col == "y_top" and deg == 2 and var == "Rz")))
            if col == "y_top" and deg == 2:
                Dmain = D
    by = pd.concat(all_df, ignore_index=True); by.to_csv(RESULTS / "E11A_by_specimen.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(RESULTS / "E11A_cross_pairs.csv", index=False)
    sm = pd.DataFrame(summ)
    m = sm[sm.is_main].iloc[0]
    case = 3 if not m.stage1 else (1 if (m.rho > 0 and m.p_one_sided < 0.05) else 2)
    sm["conclusion"] = ""
    sm.loc[sm.is_main, "conclusion"] = f"{DESC[case]} {PREMISE}"
    sm.to_csv(RESULTS / "E11A_summary.csv", index=False)

    # ---- 반복 성분 스펙트럼 (보조): 시편별 10장 평균 잔차와 짝/홀 격자 그룹별 고정 패턴
    spec_rows, mean_res = [], {s: (Dmain[s][0].mean(0), Dmain[s][1]) for s in SPECS}
    groups = {}
    for s in SPECS:
        groups.setdefault("even" if mean_res[s][1][0] % 2 == 0 else "odd", []).append(s)
    fixed = {}
    for gname, members in groups.items():
        common = sorted(set.intersection(*[set(mean_res[s][1]) for s in members]))
        fixed[gname] = (np.mean([mean_res[s][0][np.isin(mean_res[s][1], common)] for s in members], 0), np.array(common), members)

    def periodogram(name, v):
        v = v - v.mean(); n = len(v); P = np.abs(np.fft.rfft(v)) ** 2 / n
        f = np.fft.rfftfreq(n, d=2 * UM_PER_PX / 1000)  # cycles/mm (샘플 간격 2px)
        for k in range(1, len(f)):
            spec_rows.append(dict(series=name, k=k, freq_cyc_per_mm=f[k], wavelength_um=1000 / f[k], power_px2=P[k]))
        band = (f > 0.22 / (2 * UM_PER_PX / 1000)) & (f < 0.28 / (2 * UM_PER_PX / 1000))  # 8px 주기 = 0.25 cycles/샘플 부근
        return float(np.mean(P[band]) / np.median(P[1:]))
    j8 = {s: periodogram(s, mean_res[s][0]) for s in SPECS}
    j8.update({f"FIXED_{g}": periodogram(f"FIXED_{g}", v[0]) for g, v in fixed.items()})
    pd.DataFrame(spec_rows).to_csv(RESULTS / "E11A_spectrum.csv", index=False)
    pd.DataFrame([dict(series=k, band8px_over_median_power=v) for k, v in j8.items()]).to_csv(
        RESULTS / "E11A_spectrum_8px.csv", index=False)

    # ---- 재장착 대조 (NO7~NO12, poly2 윗엣지, 서술)
    rem_rows = []
    for s in ["NO7", "NO8", "NO9", "NO10", "NO11", "NO12"]:
        R, grid, _ = Dmain[s]
        loo = np.mean([np.corrcoef(np.delete(R, i, 0).mean(0), R[i])[0, 1] for i in range(10)])
        mean10 = R.mean(0); r0, rm_, lags = [], [], []
        for fn, g in remount[remount.specimen == s].groupby("filename"):
            x = g["x"].to_numpy(); r = detrend(x.astype(float), g["y_top"].to_numpy(), 2)
            v = np.interp(grid, x, r, left=np.nan, right=np.nan)
            best = (-2, 0)
            for lag in range(-5, 6):  # ±10px = ±5샘플
                a, b = (mean10[max(0, lag):len(grid) + min(0, lag)], v[max(0, -lag):len(grid) - max(0, lag)])
                ok = ~np.isnan(b)
                c = np.corrcoef(a[ok], b[ok])[0, 1]
                if lag == 0: r0.append(c)
                if c > best[0]: best = (c, lag)
            rm_.append(best[0]); lags.append(best[1])
        rem_rows.append(dict(specimen=s, r_loo9_vs1_mean=loo, r_remount_lag0_mean=np.mean(r0),
                             r_remount_maxlag_mean=np.mean(rm_), best_lag_samples_median=float(np.median(lags)), n_remount=len(r0)))
    pd.DataFrame(rem_rows).to_csv(RESULTS / "E11A_remount.csv", index=False)

    # ---- 그림
    top = by[(by.edge == "top") & (by.detrend == "poly2")].set_index("specimen")
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, s in zip(axs.ravel(), ["NO10", "NO3", "NO9", "NO8"]):
        R, grid, _ = Dmain[s]; gname = "even" if grid[0] % 2 == 0 else "odd"
        fv, fx, _ = fixed[gname]
        ax.plot(grid, R[list(COMBOS[0])].mean(0) * UM_PER_PX, lw=0.8, color="tab:blue", alpha=.7, label="반쪽 A(0~4번)")
        ax.plot(grid, R[list(SPLITS_B[0])].mean(0) * UM_PER_PX, lw=0.8, color="tab:orange", alpha=.7, label="반쪽 B(5~9번)")
        ax.plot(grid, R.mean(0) * UM_PER_PX, lw=1.6, color="k", label="10장 평균")
        ax.plot(fx, fv * UM_PER_PX, lw=1.2, ls="--", color="gray", label=f"고정 패턴({gname} 격자 시편 평균)")
        ax.set_title(f"{s}  Rz={top.at[s, 'Rz']:.1f}µm  a_spec={top.at[s, 'a_spec_um']:.1f}µm", fontsize=10)
        ax.set_xlabel("x (px)")
    for ax in axs[:, 0]: ax.set_ylabel("윗엣지 2차 추세 제거 잔차 (µm)")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGURES / "E11A_residual_examples.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(top.Rz, top.a_spec_um, s=30)
    for s, rw in top.iterrows(): ax.annotate(s, (rw.Rz, rw.a_spec_um), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Rz (µm)"); ax.set_ylabel("a_spec 윗엣지 (µm)")
    ax.set_title(f"Spearman ρ={m.rho:.2f}, 단측 순열 p={m.p_one_sided:.3f} (n=11)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "E11A_aspec_vs_Rz.png", dpi=110); plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(top[["n_cols", "interpolated", "r_AB_med", "SB_rel", "S_within_med", "S_cross_med", "n_cross_pairs", "S_spec", "a_spec_um", "p05", "has_specific", "Rz"]].round(4).to_string())
    print(sm.drop(columns="conclusion").round(4).to_string()); print(sm.loc[sm.is_main, "conclusion"].iloc[0])
    print(pd.DataFrame(rem_rows).round(3).to_string()); print({k: round(v, 2) for k, v in j8.items()})
