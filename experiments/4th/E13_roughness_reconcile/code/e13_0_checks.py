"""E13-0: 창·이미지·격자 점검 + E11 정합 점검 (거칠기와 비교하지 않음)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from paths import RESULTS, E11A_BY_SPEC
import e13_common as C

e, m, first_c = C.load()

# PREREG 2.3절 표: 1번 이미지, c, 1번 이미지 x 격자 홀짝(0=짝), 홀수 격자 이미지 수
EXPECT = {"NO2": ("20261021_153554", 8154, 0, 0), "NO3": ("20261021_153828", 8156, 1, 10),
          "NO4": ("20261021_154039", 8154, 0, 0), "NO5": ("20261021_154208", 8150, 0, 0),
          "NO6": ("20261021_154347", 8154, 0, 0), "NO7": ("20261021_154521", 8156, 1, 10),
          "NO8": ("20261021_154655", 8152, 1, 6), "NO9": ("20261021_154837", 8150, 0, 0),
          "NO10": ("20261021_155010", 8156, 1, 10), "NO11": ("20261021_155215", 8152, 1, 10),
          "NO12": ("20261021_155416", 8152, 1, 10)}


def verify(e, m, first_c):
    """입력 누락·불일치 시 결과를 쓰기 전에 실패시킨다(PREREG 2.3절 표와 대조)."""
    assert set(m.specimen) == set(C.SPECS) and (m.groupby("specimen").size() == 10).all(), "시편당 10장 아님"
    assert set(m.filename) == set(e.filename), "메타·캐시 파일 불일치"
    for s, (fn, c, par, n_odd) in EXPECT.items():
        g = m[m.specimen == s]
        assert g.iloc[0].filename.startswith(fn) and first_c[s] == c, f"{s}: 1번 이미지/c 불일치"
        pars = [C.parity(e[e.filename == f]) for f in g.filename]
        assert pars[0] == par and sum(pars) == n_odd, f"{s}: x 격자 홀짝 불일치"


verify(e, m, first_c)
imgs = {s: [(r.filename, r.c, r.idx) for r in g.itertuples()] for s, g in m.groupby("specimen")}
rows = []
# (a) 1번 이미지·c  (b) 창별 열 수  (c) 격자 홀짝
for s in C.SPECS:
    for fn, c, idx in imgs[s]:
        g = e[e.filename == fn]
        wc = C.window_C(first_c[s]); wr = C.window_Crev(first_c[s])
        xc, _ = C.cols(g, wc)
        rows.append(dict(specimen=s, idx=idx, filename=fn, c=c, c1=first_c[s],
                         n_cols_H=len(C.cols(g, C.window_H())[0]), n_pts_H=len(C.pts(g, C.window_H())[0]),
                         n_cols_C=len(xc), C_right_x=xc.max(), C_lo=wc[0], C_hi=wc[1],
                         n_cols_Crev=len(C.cols(g, wr)[0]), x_min=g.x.min(), x_max=g.x.max(), parity=C.parity(g)))
chk = pd.DataFrame(rows)

# (d) E11 정합: 윗엣지, 이미지별 캐시 전체 열, poly2
res = {s: [] for s in C.SPECS}
for s in C.SPECS:
    for fn, _, _ in imgs[s]:
        g = e[e.filename == fn]
        x = g.x.to_numpy(float)
        res[s].append((x, C.trend(x, g.y_top.to_numpy(float), "poly2")))
dec = C.decompose(res)
e11 = pd.read_csv(E11A_BY_SPEC); e11 = e11[(e11.edge == "top") & (e11.detrend == "poly2")].set_index("specimen")
d = pd.DataFrame(dec).T.join(e11[["S_within_med", "S_cross_med"]])
d["ratio_R"] = d.R / d.S_within_med; d["ratio_F"] = d.F / d.S_cross_med
k = d.drop(index="NO8")
summ = dict(rho_R_vs_Swithin=spearmanr(k.R, k.S_within_med)[0], rho_F_vs_Scross=spearmanr(k.F, k.S_cross_med)[0],
            ratio_R_median=k.ratio_R.median(), ratio_F_median=k.ratio_F.median(), n=len(k))
chk.to_csv(RESULTS / "E13_0_checks.csv", index=False)
d.index.name = "specimen"; d.to_csv(RESULTS / "E13_0_e11_reconcile.csv")
pd.DataFrame([summ]).to_csv(RESULTS / "E13_0_e11_reconcile_summary.csv", index=False)

print(chk.groupby("specimen").agg(first=("filename", "first"), c1=("c1", "first"), nH=("n_cols_H", "min"), nHmax=("n_cols_H", "max"),
      ptsH=("n_pts_H", "min"), nC=("n_cols_C", "min"), Cright=("C_right_x", "min"), nCrev=("n_cols_Crev", "min"),
      odd=("parity", "sum")).reindex(C.SPECS).to_string())
print("NO8 same-parity pairs:", sum(1 for _ in range(1)) and dec["NO8"]["n_within_pairs"])
print(d[["V1", "R", "n_within_pairs", "F", "n_cross_q", "S_within_med", "S_cross_med", "ratio_R", "ratio_F"]].round(3).to_string())
print(summ)
