"""E11-E 오차 vs 거칠기 (PREREG 6절, 반증 기록. 결과 일부를 사전에 본 확인용). 실행: /usr/bin/python3 e11e_error_vs_roughness.py"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from paths import RES1021, RESULTS, CACHE, SEED

NORMAL = [f"NO{k}" for k in range(2, 11)]
PRIOR = {("V1", "Rz"): 0.62, ("방식4", "Rz"): -0.20, ("V1", "Ra"): 0.27, ("방식4", "Ra"): 0.23}  # PREREG 0절에서 이미 본 값
N_PERM, GAIN_MIN, WINS_MIN = 10000, 2.92, 7

rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021").loc[NORMAL]
e6all = pd.read_csv(RES1021 / "E6" / "calibration_by_specimen_1021.csv", encoding="utf-8-sig")
e6all = e6all[(e6all.정의 == "cross50") & (e6all.사진선택 == "all") & (e6all.분할 == "A_시편LOSO")]
e6 = e6all[e6all.unit.isin(NORMAL)]
err = {nm: e6[e6.방식 == m].set_index("unit").loc[NORMAL] for nm, m in (("V1", "3_물리스케일고정(V1)"), ("방식4", "4_핀비율정규화선형"))}

# 촬영 순서: 본시험 첫 장 시각의 순위 (E11-B와 같은 정의)
ed = pd.read_csv(CACHE / "edges_long.csv.gz", usecols=["filename", "specimen", "단계", "시각"]).drop_duplicates("filename")
ed = ed[(ed["단계"] == "본시험(암막)") & ed.specimen.isin(NORMAL)]
t0 = ed.assign(sec=pd.to_timedelta(ed["시각"]).dt.total_seconds()).groupby("specimen").sec.min()
order = rankdata(t0.loc[NORMAL].to_numpy())


def perm_p(x, y, rng, stat):
    obs = stat(x, y)
    ps = np.array([stat(x, rng.permutation(y)) for _ in range(N_PERM)])
    return obs, float((np.sum(np.abs(ps) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1))


def sp(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def resid(v, z):  # 순위를 촬영 순서 순위에 회귀한 잔차
    r = rankdata(v); A = np.column_stack([np.ones_like(z), z])
    return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]


rng = np.random.default_rng(SEED)
rows = []
for en in ("V1", "방식4"):
    for ed_, col in (("abs", "abs_err_um"), ("signed", "err_um")):
        e = err[en][col].to_numpy()
        for rn in ("Rz", "Ra"):
            r = rough[rn].to_numpy()
            rho, p = perm_p(r, e, rng, sp)
            ex, ry = resid(e, order), resid(r, order)
            rp = float(np.corrcoef(ry, ex)[0, 1])  # 부분 ρ만 낸다(p는 잔차 직접 순열이 부적절해 뺐다, PREREG 부록 참고)
            rows.append(dict(model=en, error_def=ed_, rough=rn, n=len(e), rho=rho, p_two_sided=p,
                             rho_partial_order=rp,
                             rho_err_order=sp(e, order), rho_rough_order=sp(r, order),
                             prior_seen_rho=np.nan, matches_prior=np.nan))
# PREREG 0절에서 이미 본 값의 정의 확인(재현): NO2~NO12 n=11(NG 2개는 전체 9시편 적합의 표본 밖 예측), 부호 있는 오차.
# 지시서·PREREG 1.1절에 따라 mm 오차 분석의 주 분석은 정상 9시편이므로 이 행은 재현 확인용이다.
rough11 = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")
for en, m in (("V1", "3_물리스케일고정(V1)"), ("방식4", "4_핀비율정규화선형")):
    g = e6all[e6all.방식 == m].set_index("unit"); u = [f"NO{k}" for k in range(2, 13)]
    for rn in ("Rz", "Ra"):
        rho, p = perm_p(rough11.loc[u, rn].to_numpy(), g.loc[u, "err_um"].to_numpy(), rng, sp)
        rows.append(dict(model=en, error_def="signed(n=11 재현 확인)", rough=rn, n=len(u), rho=rho, p_two_sided=p,
                         prior_seen_rho=PRIOR[(en, rn)], matches_prior=abs(rho - PRIOR[(en, rn)]) < 0.01))
corr = pd.DataFrame(rows); corr.to_csv(RESULTS / "E11E_corr.csv", index=False)

# ---- LOSO: 방식4 vs 방식4 + g*Rz. 시편 대표값 = 모든 이미지(본시험+재장착, NO1 제외) 중앙값
vals = pd.read_csv(RES1021 / "values_by_image_1021.csv", encoding="utf-8-sig")
vals = vals[vals.specimen.isin(NORMAL)].assign(ratio16=lambda d: 16 * d.bearing_cross50_px / d.pin_cross50_px)
agg = vals.groupby("specimen").agg(true=("실측mm", "first"), ratio16=("ratio16", "median")).loc[NORMAL]
agg["Rz"] = rough.Rz
lo = []
for h in NORMAL:
    tr, te = agg.drop(h), agg.loc[h]
    X0 = np.column_stack([np.ones(len(tr)), tr.ratio16]); b0 = np.linalg.lstsq(X0, tr.true, rcond=None)[0]
    X1 = np.column_stack([X0, tr.Rz]); b1 = np.linalg.lstsq(X1, tr.true, rcond=None)[0]
    p0 = b0 @ [1, te.ratio16]; p1 = b1 @ [1, te.ratio16, te.Rz]
    lo.append(dict(fold=h, true_mm=te.true, pred_base_mm=p0, pred_rz_mm=p1, abs_err_base_um=abs(p0 - te.true) * 1000,
                   abs_err_rz_um=abs(p1 - te.true) * 1000, g_um_per_um=b1[2] * 1000, Rz=te.Rz))
lo = pd.DataFrame(lo); lo["rz_wins"] = lo.abs_err_rz_um < lo.abs_err_base_um
lo.to_csv(RESULTS / "E11E_loso.csv", index=False)
mae0, mae1 = lo.abs_err_base_um.mean(), lo.abs_err_rz_um.mean()
assert abs(mae0 - 3.99) < 0.005, f"방식4 단독 LOSO MAE {mae0:.3f} != 3.99 (재현 실패)"
gain, wins = mae0 - mae1, int(lo.rz_wins.sum())
yes = gain >= GAIN_MIN and wins >= WINS_MIN
concl = ("거칠기가 오차 일부를 설명할 가능성" if yes else "거칠기로 설명되지 않음") + " (확인용: 결과 일부를 사전에 봄)"
pd.DataFrame([dict(MAE_method4_um=mae0, MAE_method4_plus_Rz_um=mae1, gain_um=gain, rz_wins=wins, rule1_gain_ge_2_92=gain >= GAIN_MIN,
                   rule2_wins_ge_7=wins >= WINS_MIN, g_um_per_um_median=lo.g_um_per_um.median(), conclusion=concl)]).to_csv(
    RESULTS / "E11E_summary.csv", index=False)

pd.set_option("display.width", 220, "display.max_columns", 30)
print(corr.round(3).to_string(index=False)); print(lo.round(3).to_string(index=False))
print(f"MAE base {mae0:.3f} / +Rz {mae1:.3f} / gain {gain:.3f} / wins {wins}/9 -> {concl}")
