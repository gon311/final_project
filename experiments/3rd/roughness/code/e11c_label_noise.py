"""E11-C 라벨 불확도 상한과 what-if (PREREG 5절, 계산·가설 검정 없음). 실행: /usr/bin/python3 e11c_label_noise.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import RES1021, RESULTS, FIGURES, SEED

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
NORMAL = [f"NO{k}" for k in range(2, 11)]
MODES = {"0_평균찍기": "평균찍기", "3_물리스케일고정(V1)": "V1", "4_핀비율정규화선형": "방식4"}
REF = {"평균찍기": 6.64, "V1": 8.48, "방식4": 3.99}
C_GRID = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SIGMA_M = {"완벽": 0.0, "재장착 σ_m=5.2µm(원시)": 5.2, "재장착 σ_m=6.4µm(핀비율)": 6.4}
GAIN_MIN, WINS_MIN, N_ITER, N_BOOT = 2.92, 7, 10000, 2000

e6 = pd.read_csv(RES1021 / "E6" / "calibration_by_specimen_1021.csv", encoding="utf-8-sig")
e6 = e6[(e6.정의 == "cross50") & (e6.사진선택 == "all") & (e6.분할 == "A_시편LOSO") & e6.unit.isin(NORMAL)]
rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")
Rz = rough.loc[NORMAL, "Rz"].to_numpy()
true_mm = e6[e6.방식 == "0_평균찍기"].set_index("unit").loc[NORMAL, "실측mm"].to_numpy()
T = true_mm * 1000.0  # µm (엑셀 실측값 = 참값)

# ---- 재현 확인 (다르면 멈춘다)
err = {}
for m, nm in MODES.items():
    s = e6[e6.방식 == m].set_index("unit").loc[NORMAL]
    err[nm] = s.err_um.to_numpy()
    mae = np.abs(err[nm]).mean()
    assert abs(mae - REF[nm]) < 0.005, f"{nm} MAE {mae:.3f} != {REF[nm]}"

# ---- 5.1 상한: RMSE, 시편 부트스트랩 95% 구간, c_max
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(NORMAL), (N_BOOT, len(NORMAL)))
rms_rz = float(np.sqrt(np.mean(Rz ** 2))); rmsz_b = np.sqrt((Rz[idx] ** 2).mean(1))
ub = []
for nm, e in err.items():
    rmse = float(np.sqrt(np.mean(e ** 2))); rb = np.sqrt((e[idx] ** 2).mean(1))
    ub.append(dict(mode=nm, n=len(e), MAE_um=float(np.abs(e).mean()), RMSE_um=rmse,
                   RMSE_lo95=float(np.percentile(rb, 2.5)), RMSE_hi95=float(np.percentile(rb, 97.5)),
                   RMS_Rz_um=rms_rz, c_max=rmse / rms_rz,
                   c_max_lo95=float(np.percentile(rb / rmsz_b, 2.5)), c_max_hi95=float(np.percentile(rb / rmsz_b, 97.5))))
ub = pd.DataFrame(ub); ub.to_csv(RESULTS / "E11C_upper_bound.csv", index=False)

# ---- 5.2 what-if (PREREG 격자) + 사후 확장 격자(PREREG 격자 밖, 교차점 확인용)
C_EXT = [0.60, 0.70, 0.80, 1.00]
n = len(NORMAL)


def whatif(c_list, grid, rng):
    rows = []
    for c in c_list:
        eps = rng.normal(size=(N_ITER, n)) * (c * Rz)        # 라벨 잡음 µm, sd = c*Rz_i
        delta0 = rng.normal(size=(N_ITER, n))
        L = T + eps                                          # 관측 라벨
        err_mean = (L.sum(1, keepdims=True) - L) / (n - 1) - L   # 평균 찍기 LOSO 오차(관측 라벨 기준)
        mae_mean = np.abs(err_mean).mean(1)
        for name, sm in SIGMA_M.items():
            err_ins = sm * delta0 - eps                      # 예측 = 참값 + δ, 오차 = 예측 − 관측 라벨
            mae_ins = np.abs(err_ins).mean(1)
            gain = mae_mean - mae_ins; wins = (np.abs(err_ins) < np.abs(err_mean)).sum(1)
            p1, p2 = gain >= GAIN_MIN, wins >= WINS_MIN
            rows.append(dict(grid=grid, c=c, sigma_label_um=c * rms_rz, instrument=name, sigma_m_um=sm, P_rule1=p1.mean(),
                             P_rule2=p2.mean(), P_rule12=(p1 & p2).mean(), MAE_instrument_med=float(np.median(mae_ins)),
                             MAE_meanpick_med=float(np.median(mae_mean))))
    return rows


rows = whatif(C_GRID, "prereg", np.random.default_rng(SEED)) + whatif(C_EXT, "extended_post_hoc", np.random.default_rng(SEED + 1))
wi = pd.DataFrame(rows); wi.to_csv(RESULTS / "E11C_whatif.csv", index=False)
assert abs(wi[(wi.c == 0) & (wi.instrument == "완벽")].MAE_meanpick_med.iloc[0] - 6.639) < 0.005  # c=0이면 실측 평균찍기 MAE

# ---- 결론: 완벽 측정기의 P12가 50% 아래로 내려가는 c (선형 보간). PREREG 격자 안에서 먼저 찾고, 없으면 확장 격자에서 찾는다.
pf = wi[wi.instrument == "완벽"].sort_values("c").reset_index(drop=True)
pre = pf[pf.grid == "prereg"]
i = pf.index[pf.P_rule12 < 0.5]
if len(i) == 0:
    c_cross = np.nan
else:
    k = i[0]; hi, lo = pf.loc[k - 1], pf.loc[k]
    c_cross = float(hi.c + (hi.P_rule12 - 0.5) / (hi.P_rule12 - lo.P_rule12) * (lo.c - hi.c))
in_grid = bool((pre.P_rule12 < 0.5).any())
m4 = ub[ub["mode"] == "방식4"].iloc[0]
p_at_max = float(pre[pre.c == pre.c.max()].P_rule12.iloc[0])
imperf = wi[(wi.grid == "prereg") & (wi.instrument != "완벽")].groupby("instrument").P_rule12.agg(["min", "max"])
concl = (("PREREG 격자(c ≤ 0.50) 안에서는 완벽 측정기의 규칙 ①② 통과 확률이 50% 아래로 내려가지 않는다(c = 0.50에서 "
          f"{p_at_max:.0%}). 사후 확장 격자의 보간 교차점은 c ≈ {c_cross:.2f}이다. " if not in_grid else
          f"c ≥ {pre[pre.P_rule12 < 0.5].c.iloc[0]:.2f}(교차점 ≈ {c_cross:.2f})이면 완벽 측정기도 규칙 ①②를 통과할 확률이 50% 미만이다. ")
         + f"상한(5.1절)으로 본 c의 가능 범위는 ≤ {m4.c_max:.2f}(시편 부트스트랩 95% 구간 {m4.c_max_lo95:.2f}~{m4.c_max_hi95:.2f})이다. "
         + ("두 범위가 겹친다(상한 구간의 위쪽 끝만 교차점에 닿는다)." if m4.c_max_hi95 >= c_cross > m4.c_max else
            "두 범위가 겹친다." if c_cross <= m4.c_max else "두 범위가 겹치지 않는다: 라벨 잡음만으로는 완벽 측정기도 못 통과하는 크기가 상한 밖이다.")
         + f" sd = c·Rz 및 σ_m = 5.2·6.4µm(NO7~NO12 재장착 추정) 가정에서는 재장착 수준 측정기의 통과 확률이 c와 무관하게 {imperf['min'].min():.0%}~{imperf['max'].max():.0%}로 낮다. 실제 병목이 무엇인지는 미확정이다(σ_m은 정상 9시편 평가 단위로 검증하지 않았고, 잡음 모형·σ_m 민감도 분석은 하지 않았다).")
pd.DataFrame([dict(c_cross_interp=c_cross, cross_in_prereg_grid=in_grid, c_max_method4=m4.c_max,
                   c_max_lo95=m4.c_max_lo95, c_max_hi95=m4.c_max_hi95, conclusion=concl)]).to_csv(RESULTS / "E11C_conclusion.csv", index=False)

# ---- 그림
fig, ax = plt.subplots(figsize=(7, 4.6))
for (name, g), col in zip(wi.groupby("instrument", sort=False), ("tab:green", "tab:blue", "tab:orange")):
    g = g.sort_values("c"); gp = g[g.grid == "prereg"]; ge = g[g.c >= gp.c.max()]
    ax.plot(gp.c, gp.P_rule12, "o-", color=col, label=name)
    ax.plot(ge.c, ge.P_rule12, "o:", color=col, mfc="white", lw=1)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.axvspan(m4.c_max_lo95, m4.c_max_hi95, color="red", alpha=0.08)
ax.axvline(m4.c_max, color="red", ls="--", lw=1.2, label=f"c_max(방식4 RMSE/RMS(Rz)) = {m4.c_max:.2f}")
ax.set_xlabel("라벨 잡음 계수 c (sd = c·Rz)"); ax.set_ylabel("규칙 ①②를 함께 통과할 확률")
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="upper right"); ax.text(0.99, 0.02, "점선·빈 원 = 사후 확장 격자(PREREG 밖)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
top = ax.secondary_xaxis("top", functions=(lambda c: c * rms_rz, lambda s: s / rms_rz))
top.set_xlabel(f"라벨 잡음 σ (µm, c × RMS(Rz) = c × {rms_rz:.1f})")
fig.tight_layout(); fig.savefig(FIGURES / "E11C_pass_prob_vs_c.png", dpi=110); plt.close(fig)

pd.set_option("display.width", 220, "display.max_columns", 30)
print(ub.round(3).to_string(index=False)); print(wi.round(3).to_string(index=False)); print(concl)
