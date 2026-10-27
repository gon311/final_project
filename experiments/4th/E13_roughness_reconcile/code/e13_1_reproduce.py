"""E13-1: 희연님 수치 재현 (PREREG 4절). 거칠기와 비교하지 않는다."""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from paths import RESULTS, FIGURES
import e13_common as C

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
HY = {"NO2": (0.573, 0.899, 3.589), "NO9": (0.811, 1.241, 4.521), "NO10": (0.930, 1.236, 5.293)}  # MAD, STD, Range
RAWCV = {"NO2": .051, "NO3": .108, "NO4": .044, "NO5": .071, "NO6": .203, "NO7": .047, "NO8": .111,
         "NO9": .046, "NO10": .183, "NO11": .045, "NO12": .082}  # 희연님 NG1->NO11, NG2->NO12
e, m, first_c = C.load()
img1 = {s: e[e.filename == g.iloc[0].filename] for s, g in m.groupby("specimen")}
WIN = C.window_H()

rows = []
for mad, pt, tr in itertools.product(["MAD_med", "MAD_mean"], ["pts", "cols"], ["none", "poly1"]):
    for s, h in HY.items():
        x, y = (C.pts if pt == "pts" else C.cols)(img1[s], WIN)
        v = C.metrics(C.trend(x, y, tr))
        rows.append(dict(variant=f"{mad}|{pt}|{tr}", mad_def=mad, point=pt, trend=tr, specimen=s, n_points=len(x),
                         MAD_px=v[mad], STD_px=v["STD"], Range_px=v["Range"],
                         relerr_MAD=v[mad] / h[0] - 1, relerr_STD=v["STD"] / h[1] - 1, relerr_Range=v["Range"] / h[2] - 1))
var = pd.DataFrame(rows)
rep = []
for k, g in var.groupby("variant", sort=False):
    g = g.set_index("specimen").loc[list(HY)]
    ok = (g.relerr_MAD.abs() <= 0.25).all() and g.MAD_px.is_monotonic_increasing and g.MAD_px.is_unique
    rep.append(dict(variant=k, mad_def=g.mad_def.iloc[0], max_abs_relerr=g.relerr_MAD.abs().max(), reproduced=bool(ok)))
rep = pd.DataFrame(rep)
var = var.merge(rep[["variant", "reproduced"]], on="variant")
var.to_csv(RESULTS / "E13_1_variants.csv", index=False)

R = rep[rep.reproduced]
if len(R):
    best = R.sort_values(["max_abs_relerr", "mad_def"], key=lambda c: c if c.name != "mad_def" else (c != "MAD_med")).iloc[0]
    rep_mad, rep_row = best.mad_def, best.variant
    status = f"재현됨({len(R)}개 변형)"
else:
    rep_mad, rep_row, status = "MAD_med", "MAD_med|pts|none (창 H)", "엣지 추출 방식 차이로 수치 재현 불가"

# 보조: CV_diam (창 H, 10장) vs 희연님 Raw CV
cv = []
for s in C.SPECS:
    d = [C.cv_diam(e[e.filename == f], WIN) for f in m[m.specimen == s].filename]
    cv.append(dict(specimen=s, CV_diam_H=np.std(d, ddof=1) / np.mean(d) * 100, rawcv_heeyeon=RAWCV[s]))
cv = pd.DataFrame(cv)
cv["ratio"] = cv.CV_diam_H / cv.rawcv_heeyeon
cv.to_csv(RESULTS / "E13_1_cv.csv", index=False)
no9 = cv[cv.specimen != "NO9"]
summ = pd.DataFrame([dict(status=status, reproduced_variants=";".join(R.variant), representative_mad=rep_mad,
                          repro_row=rep_row, cv_rho_with_NO9=spearmanr(cv.CV_diam_H, cv.rawcv_heeyeon)[0],
                          cv_rho_without_NO9=spearmanr(no9.CV_diam_H, no9.rawcv_heeyeon)[0],
                          cv_size_ratio_median=cv.ratio.median())])
summ.to_csv(RESULTS / "E13_1_summary.csv", index=False)

# 그림: 세 시편 1번 이미지 창 H 하단 엣지
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
for a, (s, h) in zip(ax, HY.items()):
    xc, yc = C.cols(img1[s], WIN); xp, yp = C.pts(img1[s], WIN)
    vc = C.metrics(yc); a.plot(xc, yc, color="0.6", lw=1, label="cols"); a.plot(xp, yp, "o", ms=4, label="pts")
    a.axhline(np.median(yc), color="k", lw=.8, ls="--")
    vp = C.metrics(yp)
    a.set_title(f"{s}  MAD {vp['MAD_med']:.2f}/{h[0]:.2f}  STD {vp['STD']:.2f}/{h[1]:.2f}\nRange {vp['Range']:.2f}/{h[2]:.2f}  (우리/희연님, pts·none, px)", fontsize=8)
    a.set_xlabel("x (px)"); a.set_ylabel("y_bot (px)")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIGURES / "E13_1_no2_no9_no10.png", dpi=110)

pd.set_option("display.width", 200)
print(var.pivot(index="variant", columns="specimen", values="MAD_px").loc[rep.variant, list(HY)].round(3).join(rep.set_index("variant")[["max_abs_relerr", "reproduced"]].round(3)))
print(var[var.variant == "MAD_med|pts|none"][["specimen", "n_points", "STD_px", "Range_px"]].round(3).to_string())
print(summ.T.to_string()); print(cv.round(3).to_string())
