"""E13-3: 조건별 상관표 (PREREG 6·7절). 탐색. 안정군 목록 없음."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from paths import RESULTS, FIGURES, UM_PER_PX
import e13_common as C

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
s1 = pd.read_csv(RESULTS / "E13_1_summary.csv").iloc[0]
REP_MAD = s1.representative_mad
_mad, _pt, _tr = s1.repro_row.split(" ")[0].split("|") if "|" in s1.repro_row.split(" ")[0] else (REP_MAD, "pts", "none")
e, m, first_c = C.load()
rough = pd.read_csv(C.ROUGH).set_index("specimen_1021")
order = pd.Series({s: i for i, s in enumerate(C.SPECS)})
files = {s: list(g.filename) for s, g in m.groupby("specimen")}
WIN = {"H": lambda s: C.window_H(), "C": lambda s: C.window_C(first_c[s]), "C_rev": lambda s: C.window_Crev(first_c[s])}

# (이름, 창, 점, 추세, 이미지수)
ROWS = [("재현 행", "H", _pt, _tr, 1), ("H-cols-none-1", "H", "cols", "none", 1), ("C-cols-none-1", "C", "cols", "none", 1),
        ("C-cols-poly1-1", "C", "cols", "poly1", 1), ("H-cols-none-10", "H", "cols", "none", 10),
        ("C-cols-none-10", "C", "cols", "none", 10), ("C-cols-poly1-10(대표 행)", "C", "cols", "poly1", 10),
        ("C_rev-cols-poly1-10", "C_rev", "cols", "poly1", 10)]
met = []
for name, w, pt, tr, n in ROWS:
    for s in C.SPECS:
        vals = []
        for f in files[s][:n]:
            x, y = (C.pts if pt == "pts" else C.cols)(e[e.filename == f], WIN[w](s))
            vals.append(C.metrics(C.trend(x, y, tr)))
        v = pd.DataFrame(vals).median()
        met.append(dict(row=name, specimen=s, MAD=v[REP_MAD], STD=v["STD"], Range=v["Range"]))
for w in ["H", "C"]:
    for s in C.SPECS:
        d = [C.cv_diam(e[e.filename == f], WIN[w](s)) for f in files[s]]
        met.append(dict(row=f"CV_diam-{w}", specimen=s, CV_diam=np.std(d, ddof=1) / np.mean(d) * 100))
M = pd.DataFrame(met); M.to_csv(RESULTS / "E13_3_metrics_by_specimen.csv", index=False)

corr = []
for name, g in M.groupby("row", sort=False):
    g = g.set_index("specimen").loc[C.SPECS]
    for mt in [c for c in ["MAD", "STD", "Range", "CV_diam"] if g[c].notna().all()]:
        rho_o = C.perm_spearman(g[mt].to_numpy(), order.to_numpy(float))[0]
        for rv in ["Ra", "Rq", "Rz"]:
            r = rough[rv].reindex(C.SPECS).to_numpy()
            rho, p = C.perm_spearman(g[mt].to_numpy(), r)
            corr.append(dict(row=name, metric=mt, rough_var=rv, n=len(g), spearman_rho=rho, p_two_sided=p,
                             pearson_r=pearsonr(g[mt], r)[0], rho_order=rho_o))
K = pd.DataFrame(corr); K.to_csv(RESULTS / "E13_3_corr.csv", index=False)

# 결론 문장 (PREREG 7절)
rep_name, rep_main = "재현 행", "C-cols-poly1-10(대표 행)"
r_rep = K[(K.row == rep_name) & (K.metric == "MAD") & (K.rough_var == "Rq")].iloc[0]
rep_row = K[(K.row == rep_main) & (K.metric == "MAD") & (K.rough_var == "Rq")].iloc[0]
d2 = pd.read_csv(RESULTS / "E13_2_summary.csv").iloc[0]
pos = r_rep.pearson_r >= 0.30
rho_S = d2.rho_sqrtSpos_Rq
if not pos:
    case, sent = "B", "희연님의 양의 상관은 우리 엣지 데이터(전체 11시편)로 재현되지 않는다. 차이는 엣지 추출·창·시편 선택의 차이로 보인다(탐색, n = 11)."
elif rho_S <= 0:
    case, sent = "A", "희연님의 양의 상관은 우리 엣지 데이터에서도 보이지만, 그 상관은 시편 고유 표면 성분이 아니라 한 장 산포의 다른 성분에서 나온다. 거칠기보다 측정 쪽 변동이 원인일 가능성(탐색, n = 11)."
else:
    case, sent = "C", "한 장 산포와 시편 고유 성분 모두 Rq와 같은 방향이다. E11-A 하단 결과(ρ −0.58)와 상충하며 원인은 이 데이터로 판단 불가(탐색, n = 11)."
comp = f"1번 이미지 하단 엣지 산포의 주성분은 {d2.main_component}이다(11시편 몫 중앙값 S {d2.med_share_S:.2f}, F {d2.med_share_F:.2f}, N {d2.med_share_N:.2f})."
fixed = "E11의 결론(주 가설 문장 2번)은 바뀌지 않는다. E13은 새 검정이 아니라 재현·조정이다."
pd.DataFrame([dict(rep_row=f"{rep_name}({_pt},{_tr},{REP_MAD})", rep_row_pearson_r_Rq=r_rep.pearson_r, rep_row_rho_Rq=r_rep.spearman_rho,
                   rep_row_p=r_rep.p_two_sided, representative_row=rep_main, representative_rho_Rq=rep_row.spearman_rho,
                   representative_p=rep_row.p_two_sided, representative_r_Rq=rep_row.pearson_r, rho_sqrtSpos_Rq=rho_S,
                   case=case, conclusion=sent, component_sentence=comp, fixed_sentence=fixed,
                   stable_group="안정군 목록 없음")]).to_csv(RESULTS / "E13_3_summary.csv", index=False)

# 그림
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, name in zip(ax, [rep_name, rep_main]):
    g = M[M.row == name].set_index("specimen").loc[C.SPECS]; k = K[(K.row == name) & (K.metric == "MAD") & (K.rough_var == "Rq")].iloc[0]
    sc = a.scatter(rough.Rq.reindex(C.SPECS), g.MAD * UM_PER_PX, c=order.to_numpy(), cmap="viridis")
    for s in C.SPECS:
        a.annotate(s, (rough.Rq[s], g.MAD[s] * UM_PER_PX), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("Rq (µm)"); a.set_ylabel("MAD (µm)"); a.set_title(f"{name}\nρ {k.spearman_rho:+.2f}, p {k.p_two_sided:.2f}, r {k.pearson_r:+.2f}", fontsize=9)
fig.colorbar(sc, ax=ax, label="촬영 순서"); fig.savefig(FIGURES / "E13_3_rep_vs_Rq.png", dpi=110, bbox_inches="tight")

pd.set_option("display.width", 220)
print(K[K.rough_var == "Rq"].round(3).to_string(index=False)); print(case, sent); print(comp)
print(K[(K.metric == "MAD") & (K.rough_var != "Rq")][["row", "rough_var", "spearman_rho", "pearson_r"]].round(2).to_string(index=False))
