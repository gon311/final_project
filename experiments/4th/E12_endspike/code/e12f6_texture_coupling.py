"""E12-F6: 엣지 위치 편차가 엣지 바로 바깥 밝은 배경의 국소 밝기(질감)로 예측되는가. 학습은 8240~8290 열을 뺀 열로, 8265 부근은 홀드아웃.
10/21 본시험 NO2~NO12만. 규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 8절(실행 전 고정)."""
import numpy as np
import pandas as pd
from paths import RESULTS, FIGURES, PROFILES as CACHE
from e12_edges import main, load

X0, X1, HALF, LEVEL = 6000, 11000, 150, 105
HO = (8240, 8290)                        # 홀드아웃 열 구간(포함)
PL = np.arange(12, 31)                   # 엣지에서 12~30px 바깥의 밝은 평탄부(변환 영역 밖)
R_MIN, K_MIN, RATIO = 0.3, 6, (0.5, 1.5)
C_MIN, FRAC = 0.5, 8 / 11
XS = np.arange(X0, X1)


def slab_arrays(sub, edge):
    """sub (rows, X1-X0): 열마다 엣지 y(서브픽셀)와 엣지 바깥 평탄부 평균 밝기 L. 엣지를 못 찾거나 인덱스가 범위 밖이면 NaN."""
    hit = (sub[:-1] >= LEVEL) & (sub[1:] < LEVEL) if edge == "top" else (sub[:-1] < LEVEL) & (sub[1:] >= LEVEL)
    has, k = hit.any(0), hit.argmax(0); col = np.arange(sub.shape[1])
    v0, v1 = sub[k, col], sub[np.minimum(k + 1, sub.shape[0] - 1), col]
    frac = (v0 - LEVEL) / (v0 - v1) if edge == "top" else (LEVEL - v0) / (v1 - v0)
    ye = np.where(has, k + frac, np.nan)
    idx = k[None, :] - PL[:, None] if edge == "top" else k[None, :] + PL[:, None]
    ok = has & (idx.min(0) >= 0) & (idx.max(0) < sub.shape[0])
    L = np.where(ok, sub[np.clip(idx, 0, sub.shape[0] - 1), col[None, :]].mean(0), np.nan)
    return ye, L


def detr(a):
    return a - pd.Series(a).rolling(15, center=True, min_periods=15).median().values


def fit_one(dev, dL, specimen_sign):
    """한 시편(장 여러 개)에서 학습 열로 dev = β·dL 과 r을 구하고 홀드아웃에서 예측. dev, dL: (n_img, n_x)."""
    tr = ~np.isnan(dev).any(0) & ~np.isnan(dL).any(0) & ~((XS >= HO[0]) & (XS <= HO[1]))
    d, l = dev[:, tr].ravel(), dL[:, tr].ravel()
    beta = float((d * l).sum() / (l ** 2).sum()); r = float(np.corrcoef(d, l)[0, 1])
    ho = (XS >= HO[0]) & (XS <= HO[1]) & ~np.isnan(dev).any(0) & ~np.isnan(dL).any(0)
    obs, pred = np.median(dev[:, ho], 0), np.median(beta * dL[:, ho], 0)
    t = np.isin(XS[ho], (8265, 8266))
    o, p = float(obs[t].mean()), float(pred[t].mean())
    return dict(beta=beta, r=r, n_train=int(tr.sum()), obs_8265_6=o, pred_8265_6=p, ratio=o / p if p != 0 else np.nan,
                ho_corr=float(np.corrcoef(obs, pred)[0, 1]) if ho.sum() > 5 else np.nan, sign_ok=bool(np.sign(r) == specimen_sign))


def verdict(B):
    top, bot = B[B.edge == "top"], B[B.edge == "bot"]
    sg = lambda q: np.where(q.edge == "top", 1, -1)       # 예측 부호: 윗엣지 +, 아랫엣지 −
    a = all(((q.r * sg(q)) >= R_MIN).sum() >= np.ceil(FRAC * len(q)) for q in (top, bot))
    sp = top[top.obs_8265_6 <= -0.5]
    ok = sp.ratio.between(*RATIO) & (np.sign(sp.obs_8265_6) == np.sign(sp.pred_8265_6))
    b = len(sp) >= K_MIN and ok.sum() >= np.ceil(2 * len(sp) / 3)
    c = (bot.ho_corr >= C_MIN).sum() >= np.ceil(FRAC * len(bot))
    why = [n for n, v in (("(a) 결합 r ≥ 0.3·예측 부호", a), ("(b) 8265 예측 비율", b), ("(c) 아랫엣지 홀드아웃 상관", c)) if not v]
    if not why:
        return "엣지 위치 편차가 엣지 바깥 배경 밝기(질감)로 예측됨 — 8265 스파이크는 배경 질감과 결합한 측정 편향 후보, 센서·렌즈·ISP 결함 가설에 불리, 원인 판단 보류"
    return "판단 불가 (미충족: " + ", ".join(why) + ")"


if __name__ == "__main__":
    z = np.load(CACHE, allow_pickle=True)
    cen = {f: (int(a), int(b)) for f, a, b in zip(z["filename"].astype(str), z["top_center"], z["bot_center"])}
    rows = [r for r in main.itertuples() if r.specimen != "NO1"]
    assert all(r.filename in cen for r in rows) and len(rows) == 110, "입력 불완전"
    acc = {}
    for r in rows:
        g = load(r)
        for ei, edge in enumerate(("top", "bot")):
            c = cen[r.filename][ei]
            ye, L = slab_arrays(g[c - HALF:c + HALF + 1, X0:X1], edge)
            acc.setdefault((r.specimen, edge), []).append((detr(ye), detr(L)))
    recs, curves = [], []
    for (s, edge), L in acc.items():
        dev, dL = np.stack([a for a, _ in L]), np.stack([b for _, b in L])
        recs.append(dict(specimen=s, edge=edge, n_img=len(L), **fit_one(dev, dL, +1 if edge == "top" else -1)))
        if s in ("NO10", "NO5", "NO9"):
            m = (XS >= 8200) & (XS <= 8330)
            beta = recs[-1]["beta"]
            curves += [dict(specimen=s, edge=edge, x=int(x), obs=float(o), pred=float(beta * p)) for x, o, p in
                       zip(XS[m], np.nanmedian(dev[:, m], 0), np.nanmedian(dL[:, m], 0))]
    B = pd.DataFrame(recs).round(3); B.to_csv(RESULTS / "E12F6_by_specimen.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "E12F6_curves.csv", index=False)
    v = verdict(B); pd.DataFrame([dict(verdict=v)]).to_csv(RESULTS / "E12F6_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(B.sort_values(["edge", "specimen"]).to_string(index=False)); print("판정:", v)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = pd.DataFrame(curves); fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    for j, s in enumerate(("NO10", "NO5", "NO9")):
        for i, edge in enumerate(("top", "bot")):
            q = C[(C.specimen == s) & (C.edge == edge)]; a = ax[i, j]
            a.plot(q.x, q.obs, label="observed edge dev"); a.plot(q.x, q.pred, label="predicted from bg texture"); a.axvline(8265, c="r", ls=":")
            a.set_title(f"{s} {edge}"); a.set_ylabel("px")
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "E12F6_texture_coupling.png", dpi=110)
