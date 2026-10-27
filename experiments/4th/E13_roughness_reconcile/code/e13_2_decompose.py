"""E13-2: 한 장 산포 성분 분해 (PREREG 5절). V1 = R + N, R = S + F."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import RESULTS, FIGURES
import e13_common as C

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
REP_MAD = pd.read_csv(RESULTS / "E13_1_summary.csv").representative_mad.iloc[0]
SETTINGS = [("C", "poly2"), ("C", "poly1"), ("H", "poly2")]  # 첫째가 주 설정
e, m, first_c = C.load()
rough = pd.read_csv(C.ROUGH).set_index("specimen_1021")
order = {s: i for i, s in enumerate(C.SPECS)}

rows, links, summ = [], [], []
for win_name, tr in SETTINGS:
    res, mad1 = {}, {}
    for s in C.SPECS:
        win = C.window_H() if win_name == "H" else C.window_C(first_c[s])
        res[s] = []
        for f in m[m.specimen == s].filename:
            x, y = C.cols(e[e.filename == f], win)
            res[s].append((x, C.trend(x, y, tr)))
        mad1[s] = C.metrics(res[s][0][1])[REP_MAD]
    dec = C.decompose(res)
    d = pd.DataFrame(dec).T
    d["S"] = d.R - d.F; d["N"] = d.V1 - d.R
    for k in "SFN":
        d[f"share_{k}"] = d[k] / d.V1
    d["n_cols"] = [len(res[s][0][0]) for s in d.index]
    d["MAD_img1"] = pd.Series(mad1)
    for r in ["Rq", "Ra", "Rz"]:
        d[r] = rough[r].reindex(d.index)
    d["order"] = [order[s] for s in d.index]
    d.insert(0, "setting", f"{win_name}-{tr}"); d.index.name = "specimen"
    rows.append(d.reset_index())
    comp = {"sqrtSpos": np.sqrt(np.maximum(d.S, 0)), "sqrtF": np.sqrt(d.F), "sqrtN": np.sqrt(d.N)}
    for cn, cv in comp.items():
        for tn, tv in [("MAD_img1", d.MAD_img1), ("Rq", d.Rq), ("Ra", d.Ra), ("Rz", d.Rz), ("order", d.order)]:
            rho, p = C.perm_spearman(cv.to_numpy(), tv.to_numpy(float))
            links.append(dict(setting=f"{win_name}-{tr}", comp=cn, target=tn, rho=rho, p_two_sided=p))
    med = {k: float(d[f"share_{k}"].median()) for k in "SFN"}
    main = [k for k in "SFN" if med[k] >= 0.5]
    name = {"S": "시편 고유", "F": "고정 패턴", "N": "장마다 다른 잡음"}
    summ.append(dict(setting=f"{win_name}-{tr}", med_share_S=med["S"], med_share_F=med["F"], med_share_N=med["N"],
                     main_component=name[main[0]] if main else "혼합",
                     rho_sqrtSpos_Rq=[l for l in links if l["setting"] == f"{win_name}-{tr}" and l["comp"] == "sqrtSpos" and l["target"] == "Rq"][0]["rho"],
                     n_S_negative=int((d.S < 0).sum())))
by = pd.concat(rows); lk = pd.DataFrame(links); sm = pd.DataFrame(summ)
by.to_csv(RESULTS / "E13_2_by_specimen.csv", index=False)
lk.to_csv(RESULTS / "E13_2_links.csv", index=False)
sm.to_csv(RESULTS / "E13_2_summary.csv", index=False)

# 그림: 주 설정, Rq 순서, S·F·N 몫 누적 막대 (S<0은 0 아래)
d = by[by.setting == "C-poly2"].sort_values("Rq")
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(d)); Sp = d.share_S.clip(lower=0)
ax.bar(xs, d.share_F, label="F 고정 패턴", color="0.65")
ax.bar(xs, d.share_N, bottom=d.share_F, label="N 장마다 잡음", color="tab:orange")
ax.bar(xs, Sp, bottom=d.share_F + d.share_N, label="S 시편 고유", color="tab:blue")
ax.bar(xs, d.share_S.clip(upper=0), color="tab:blue", hatch="//", alpha=.6)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(xs); ax.set_xticklabels([f"{s}\nRq {r:.2f}" for s, r in zip(d.specimen, d.Rq)], fontsize=7)
ax.set_ylabel("V1 대비 몫"); ax.set_title("E13-2 주 설정(창 C, poly2, 하단): 한 장 산포 성분 몫 (Rq 오름차순)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "E13_2_shares.png", dpi=110)

pd.set_option("display.width", 220)
print(by[by.setting == "C-poly2"][["specimen", "n_cols", "n_within_pairs", "n_cross_q", "V1", "R", "F", "S", "N", "share_S", "share_F", "share_N", "MAD_img1", "Rq"]].round(3).to_string(index=False))
print(sm.round(3).to_string(index=False))
print(lk[(lk.setting == "C-poly2")].round(3).to_string(index=False))
