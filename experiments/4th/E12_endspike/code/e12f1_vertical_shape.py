"""E12-F1: 8265 스파이크의 수직 구조. 엣지 변환 영역에서만(엣지 변위) 나타나는가, 열 전체 밝기에 걸치는가(밝기 결함).
10/21 본시험 NO2~NO12만 사용. 규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 7절(실행 전 고정)."""
import numpy as np
import pandas as pd
from paths import RESULTS, FIGURES, PROFILES as CACHE
from e12_edges import main, load

TX, HALF, FITW, LEVEL, N_MIN_SPEC = 8265, 380, 30, 105, 6
XP = [x for x in range(8040, 8489, 8) if abs(x - TX) > 24]       # 가짜 목표(8열 간격, 진짜 목표 ±24 제외)
U = {"top": np.arange(-300, 61), "bot": np.arange(-60, 301)}      # 엣지 기준 y 격자
FLAT = {"top": (-300, -120), "bot": (120, 300)}                   # 밝은 평평 영역(엣지에서 120px 이상)
EDGES = ("top", "bot")


def cross(col, edge):
    hit = (col[:-1] >= LEVEL) & (col[1:] < LEVEL) if edge == "top" else (col[:-1] < LEVEL) & (col[1:] >= LEVEL)
    k = np.flatnonzero(hit)
    return None if not len(k) else k[0] + (col[k[0]] - LEVEL) / (col[k[0]] - col[k[0] + 1])


def image_profiles(g, c, edge, xt):
    """목표 열 {xt, xt+1}과 양옆 열(xt-8..xt-4, xt+5..xt+9)의 수직 프로파일을 엣지 위치 기준으로 정렬해 (Δ, F)를 낸다."""
    sub = g[c - HALF:c + HALF + 1, xt - 8:xt + 10]
    T = sub[:, 8:10].mean(1)
    F = np.concatenate([sub[:, 0:5], sub[:, 13:18]], 1).mean(1)
    ye = cross(F[HALF - 60:HALF + 61], edge)
    if ye is None: return None
    y = ye + HALF - 60 + U[edge]; r = np.arange(len(F))
    f = np.interp(y, r, F)
    return np.interp(y, r, T) - f, f


def fit(delta, F, edge):
    """엣지 변위 모델 Δ ≈ -s·F'(u). s(px, 부호: +y 방향), R², 평평 영역 평균 Δ, 모델 엣지 진폭 A, ρ = |Δ_flat|/A."""
    u = U[edge]; Fp = np.gradient(F); m = np.abs(u) <= FITW
    s = -float((delta[m] * Fp[m]).sum() / (Fp[m] ** 2).sum())
    sse = float(((delta[m] + s * Fp[m]) ** 2).sum()); sst = float((delta[m] ** 2).sum())
    lo, hi = FLAT[edge]; flat = float(delta[(u >= lo) & (u <= hi)].mean())
    A = abs(s) * float(np.abs(Fp[m]).max())
    return dict(s_px=s, r2=1 - sse / sst if sst > 0 else np.nan, flat_gray=flat, A_gray=A, rho=abs(flat) / A if A > 0 else np.nan)


def classify(r2, rho):
    if r2 >= 0.8 and rho <= 0.25: return "edge"      # 엣지 변위로 설명, 열 전체 밝기 변화 작음
    if rho >= 0.5: return "column"                    # 평평 영역의 밝기 변화가 엣지 모델 진폭의 절반 이상
    return "mixed"


def verdict(by_top):
    pres = by_top[by_top.present]
    if len(pres) < N_MIN_SPEC: return f"판단 불가 (스파이크 재현 시편 {len(pres)} < {N_MIN_SPEC})"
    vc = pres["class"].value_counts(); k, n = vc.index[0], len(pres)
    if vc.iloc[0] < np.ceil(2 * n / 3): return f"판단 불가 (유형 분산: {vc.to_dict()})"
    return {"edge": "엣지 변환 영역의 변위로 설명됨 — 열 전체 밝기 결함이 아님, 원인 판단 보류",
            "column": "열 전체 밝기 변화로 설명됨 — 엣지 변위가 아님, 원인 판단 보류",
            "mixed": "혼합 — 엣지 변위와 열 밝기 변화가 함께 있음, 원인 판단 보류"}[k]


if __name__ == "__main__":
    z = np.load(CACHE, allow_pickle=True)
    cen = {f: (int(a), int(b)) for f, a, b in zip(z["filename"].astype(str), z["top_center"], z["bot_center"])}
    rows = [r for r in main.itertuples() if r.specimen != "NO1"]
    assert all(r.filename in cen for r in rows) and len(rows) == 110, "입력 불완전"
    acc = {}
    for r in rows:
        g = load(r)
        for ei, edge in enumerate(EDGES):
            for xt in [TX] + XP:
                out = image_profiles(g, cen[r.filename][ei], edge, xt)
                if out is not None:
                    acc.setdefault((r.specimen, edge, xt), []).append(out)
    recs, prof = [], []
    for (s, edge, xt), L in acc.items():
        d = np.median([a for a, _ in L], 0); f = np.median([b for _, b in L], 0)
        recs.append(dict(specimen=s, edge=edge, x=xt, n_img=len(L), **fit(d, f, edge)))
        if xt == TX: prof += [dict(specimen=s, edge=edge, u=int(u), delta=float(a), F=float(b)) for u, a, b in zip(U[edge], d, f)]
    R = pd.DataFrame(recs); pd.DataFrame(prof).to_csv(RESULTS / "E12F1_profiles.csv", index=False)
    by = []
    for (s, edge), q in R.groupby(["specimen", "edge"]):
        t = q[q.x == TX].iloc[0]; ps = q[q.x != TX].s_px.abs()
        p = (1 + int((ps >= abs(t.s_px)).sum())) / (1 + len(ps))
        by.append(dict(specimen=s, edge=edge, n_img=int(t.n_img), s_px=round(t.s_px, 2), s_um=round(t.s_px * 26.092), r2=round(t.r2, 3),
                       flat_gray=round(t.flat_gray, 2), A_gray=round(t.A_gray, 1), rho=round(t.rho, 3), n_pseudo=len(ps),
                       pseudo_abs_s_median=round(float(ps.median()), 2), p_rank=round(p, 4), present=bool(p <= 0.05),
                       **{"class": classify(t.r2, t.rho)}))
    B = pd.DataFrame(by); B.to_csv(RESULTS / "E12F1_by_specimen.csv", index=False)
    top, bot = B[B.edge == "top"], B[B.edge == "bot"]
    v = verdict(top)
    sg = pd.merge(top[["specimen", "s_px"]], bot[["specimen", "s_px"]], on="specimen", suffixes=("_top", "_bot"))
    same = int((np.sign(sg.s_px_top) == np.sign(sg.s_px_bot)).sum())
    pd.DataFrame([dict(verdict=v, n_top_present=int(top.present.sum()), n_bot_present=int(bot.present.sum()),
                       sign_same_top_bot=f"{same}/{len(sg)}")]).to_csv(RESULTS / "E12F1_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(B.sort_values(["edge", "specimen"]).to_string(index=False)); print("판정:", v, "| 윗·아랫 s 부호 같은 시편", f"{same}/{len(sg)}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = pd.DataFrame(prof); fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, edge in zip(ax, EDGES):
        for s, q in P[P.edge == edge].groupby("specimen"): a.plot(q.u, q.delta, lw=.8, label=s)
        a.axhline(0, c="k", lw=.5); a.set_title(f"{edge} edge: Δ(u)=I(8265,8266)−flank"); a.set_xlabel("u = y − edge (px)"); a.set_ylabel("gray")
    ax[0].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(FIGURES / "E12F1_vertical_shape.png", dpi=110)
