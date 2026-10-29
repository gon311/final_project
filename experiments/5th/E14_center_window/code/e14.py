"""E14 중앙 구간 대표값 (docs/PREREG_E14.md, docs/EXPERIMENTS_E14.md).

python e14.py --step 0   # 데이터 준비, 재현 확인, 검정 구현 점검, 검정력 모의실험
"""
import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from paths import EDGES, E6_CODE, RES1021, RESULTS, SEED

sys.path.insert(0, str(E6_CODE))
import calibration as C  # noqa: E402  (원본, 수정 안 함)

RR = json.load(open(RES1021 / "repeatability_ratio.json"))
SCALE = RR["V1_scale_mm_per_px"]  # 0.026092 mm/px
C.PHYSICAL_SCALE = SCALE  # run_e6_1021.py와 같은 덮어쓰기
NORMAL = ["NO%d" % k for k in range(2, 11)]
W2_MM, W3_MM = 4.5 / 30 * 10.2, 2 / 30 * 10.2  # 중앙 10·5개소 상당 반폭 (PREREG 2절)
BIN_MM = 0.1
E6_MAE = {"0_평균찍기": 6.638889, "3_물리스케일고정(V1)": 8.478581, "4_핀비율정규화선형": 3.991198}
CUT_1, CUT_2 = 9 / 512, 20 / 512  # 보정 Holm 기준(명목 0.025, 0.05), PREREG 5.2·부록 A-2
PATTERNS = np.array(list(itertools.product([0, 1], repeat=len(NORMAL))), dtype=bool)  # 512 x 9


def robust_fit(x, y):
    """measure.py `_measure_window`의 robust_fit과 같은 산술."""
    p = np.polyfit(x, y, 1)
    r = y - np.polyval(p, x)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    k = np.abs(r) < 4 * 1.4826 * mad
    return np.polyfit(x[k], y[k], 1)


def load():
    e = pd.read_csv(EDGES)
    v = pd.read_csv(RES1021 / "values_by_image_1021.csv", encoding="utf-8-sig")
    v = v[v.specimen != "NO1"]
    cols = ["filename", "실측mm", "pin_cross50_px", "bearing_cross50_px"]
    m = e.merge(v[cols], on="filename", how="left", validate="m:1")
    assert m["실측mm"].notna().all(), "values_by_image에 없는 filename"
    assert set(e.filename) == set(v.filename), "edges와 values의 이미지 집합이 다름"
    return m


def per_image(g):
    """이미지 한 장: R0(d_line), θ, xm, 열 직경 d(x)."""
    x = g.x.to_numpy(float)
    pt, pb = robust_fit(x, g.y_top.to_numpy()), robust_fit(x, g.y_bot.to_numpy())
    xc = x.mean()
    th = np.arctan(0.5 * (pt[0] + pb[0]))
    r0 = (np.polyval(pb, xc) - np.polyval(pt, xc)) * np.cos(th)
    d = (g.y_bot.to_numpy() - g.y_top.to_numpy()) * np.cos(th)
    xm = 0.5 * (x.min() + x.max())
    return r0, th, xm, x, d


def specimen_table(img, col):
    """시편 대표값 = 이미지 중앙값 (E6 규칙). 방식 4용 ratio도 이미지별로 나눈 뒤 중앙값."""
    t = img.assign(ratio=img[col] / img["pin_cross50_px"])
    return t.groupby("specimen").agg(실측mm=("실측mm", "first"), px=(col, "median"), ratio=("ratio", "median"))


def loso_err(tab, fitter, kind="px"):
    """정상 9시편 LOSO, 시편별 오차(µm)."""
    t = tab.loc[NORMAL, ["실측mm", kind]].rename(columns={kind: "px"})
    out = []
    for held in NORMAL:
        fn, _ = fitter(t.drop(index=held))
        out.append((fn(float(t.loc[held, "px"])) - t.loc[held, "실측mm"]) * 1000)
    return np.array(out)


def mae_m3(px, truth):
    """방식 3 LOSO MAE(µm). calibration.fit_fixed_scale을 그대로 거친다."""
    tab = pd.DataFrame({"실측mm": truth, "px": px}, index=NORMAL)
    return float(np.abs(loso_err(tab, C.fit_fixed_scale)).mean())


def swap_test(a_px, b_px, truth):
    """PREREG 5.1: 시편 내 A<->B 교환 무작위화 검정. 패턴마다 보정·LOSO 전체를 다시 한다.
    반환: T_관측(=MAE_B−MAE_A), 단측 p(B가 더 작은 쪽), 패턴별 T."""
    a, b = a_px - a_px.mean(), b_px - b_px.mean()
    ts = np.empty(len(PATTERNS))
    for k, s in enumerate(PATTERNS):
        aa, bb = np.where(s, b, a), np.where(s, a, b)
        ts[k] = mae_m3(bb, truth) - mae_m3(aa, truth)
    t_obs = ts[0]  # 패턴 0 = 교환 없음
    return t_obs, float(np.mean(ts <= t_obs + 1e-12)), ts


def loso_mae_z(z):
    """방식 3 LOSO MAE의 닫힌 식: e_i = (9/8)(z_i − z̄). z: (..., 9) µm."""
    n = z.shape[-1]
    return np.abs(n / (n - 1) * (z - z.mean(-1, keepdims=True))).mean(-1)


def swap_ts_z(za, zb):
    """swap_test의 z 공간 판: px 중심화 = z를 각자 표본평균으로 중심화(공통 상수 mean(실측)은 MAE 불변).
    중심화 없이 교환하면 부분 교환에서 두 평균 차이가 섞여 다른 검정이 된다(Codex 리뷰 2)."""
    za = za - za.mean(-1, keepdims=True)
    zb = zb - zb.mean(-1, keepdims=True)
    s = PATTERNS[None]
    aa = np.where(s, zb[:, None], za[:, None])
    bb = np.where(s, za[:, None], zb[:, None])
    return loso_mae_z(bb) - loso_mae_z(aa)  # reps x 512


def power_sim(reps=2000):
    """PREREG 4.1. z 공간 시뮬레이션, swap_ts_z로 실제 swap_test와 같은 검정을 한다."""
    rng = np.random.default_rng(SEED)
    n = len(NORMAL)
    k = n / (n - 1) * np.sqrt((n - 1) / n) * np.sqrt(2 / np.pi)  # E[MAE] = k·SD
    sd_a = E6_MAE["3_물리스케일고정(V1)"] / k
    rows = []
    for dlt in (0, 1, 3, 5):
        sd_b = sd_a * (E6_MAE["3_물리스케일고정(V1)"] - dlt) / E6_MAE["3_물리스케일고정(V1)"]
        for rho in (0.5, 0.8, 0.95):
            cov = [[sd_a ** 2, rho * sd_a * sd_b], [rho * sd_a * sd_b, sd_b ** 2]]
            ab = rng.multivariate_normal([0, 0], cov, size=(reps, n))  # reps x 9 x 2
            za, zb = ab[..., 0], ab[..., 1]
            ts = swap_ts_z(za, zb)
            p = (ts <= ts[:, :1] + 1e-12).mean(1)
            rows.append(dict(MAE감소_um=dlt, rho=rho, SD_A_um=sd_a, SD_B_um=sd_b, reps=reps,
                             기각률_a05=float((p < 0.05).mean()), 기각률_a025=float((p < 0.025).mean()),
                             기각률_보정20of512=float((p <= CUT_2 + 1e-12).mean()),
                             기각률_보정9of512=float((p <= CUT_1 + 1e-12).mean()),
                             평균MAE감소_모의=float((loso_mae_z(za) - loso_mae_z(zb)).mean())))
    return pd.DataFrame(rows)


def step0():
    m = load()
    checks, imgs, bins = [], [], []
    for f, g in m.groupby("filename", sort=False):
        r0, th, xm, x, d = per_image(g)
        dx_mm = (x - xm) * SCALE
        imgs.append(dict(filename=f, specimen=g.specimen.iat[0], mount=int(g["mount"].iat[0]),
                         실측mm=g["실측mm"].iat[0], pin_cross50_px=g.pin_cross50_px.iat[0],
                         R0=r0, e6=g.bearing_cross50_px.iat[0],
                         n_W1=len(x), n_W2=int((np.abs(dx_mm) <= W2_MM).sum()), n_W3=int((np.abs(dx_mm) <= W3_MM).sum())))
        bins.append(pd.DataFrame(dict(specimen=g.specimen.iat[0], filename=f, b=np.floor(dx_mm / BIN_MM).astype(int))))
    img = pd.DataFrame(imgs)

    # (a) R0 재현
    diff = (img.R0 - img.e6).abs()
    checks.append(dict(항목="a_R0재현_max_px", 값=diff.max(), 기준="<0.01", 통과=bool(diff.max() < 0.01)))

    # (b) E6 LOSO 재현 (R0, all)
    tab = specimen_table(img, "R0")
    for name, fitter, kind in [("0_평균찍기", C.fit_const, "px"), ("3_물리스케일고정(V1)", C.fit_fixed_scale, "px"),
                               ("4_핀비율정규화선형", C.fit_linear, "ratio")]:
        mae = float(np.abs(loso_err(tab, fitter, kind)).mean())
        checks.append(dict(항목=f"b_E6재현_{name}_MAE_um", 값=mae, 기준=f"{E6_MAE[name]}±0.01",
                           통과=bool(abs(mae - E6_MAE[name]) < 0.01)))

    # (c) 창 안 열 수, 시편별 공통 구간 수
    for w in ("n_W1", "n_W2", "n_W3"):
        checks.append(dict(항목=f"c_{w}_min/max", 값=f"{img[w].min()}/{img[w].max()}", 기준="W3≥20" if w == "n_W3" else "",
                           통과=bool(img[w].min() >= 20) if w == "n_W3" else None))
    bb = pd.concat(bins).drop_duplicates()
    nimg = bb.groupby("specimen").filename.nunique()
    common = bb.groupby(["specimen", "b"]).filename.nunique().reset_index()
    common = common[common.filename == common.specimen.map(nimg)].groupby("specimen").size()
    checks.append(dict(항목="c_시편별_공통0.1mm구간수_min/max", 값=f"{common.min()}/{common.max()}", 기준="", 통과=None))

    # (d) 검정 구현 점검
    truth = tab.loc[NORMAL, "실측mm"].to_numpy()
    px0 = tab.loc[NORMAL, "px"].to_numpy()
    t, p, _ = swap_test(px0, px0, truth)
    checks.append(dict(항목="d_A=B일때_T,p", 값=f"{t:.2e},{p:.3f}", 기준="T=0, p=1", 통과=bool(abs(t) < 1e-9 and p == 1)))
    # 가짜 후보(R0 + 잡음)로 '전부 교환' T = −T_관측 확인 — 실제 후보는 아직 계산하지 않는다
    fake = px0 + np.random.default_rng(SEED).normal(0, 0.2, len(px0))
    t, _, ts = swap_test(px0, fake, truth)
    checks.append(dict(항목="d_전부교환_T=-T관측", 값=abs(ts[-1] + t), 기준="<1e-9", 통과=bool(abs(ts[-1] + t) < 1e-9)))
    z = (SCALE * px0 - truth) * 1000
    closed = float(loso_mae_z(z))
    checks.append(dict(항목="d_닫힌식_vs_calibration_um", 값=abs(closed - mae_m3(px0, truth)), 기준="<1e-9",
                       통과=bool(abs(closed - mae_m3(px0, truth)) < 1e-9)))
    # 모의실험 검정(swap_ts_z) = 실제 swap_test 회귀 검증: 모의 z를 px로 바꿔 패턴별 T·p 비교 (Codex 리뷰 2)
    rng = np.random.default_rng(SEED)
    worst_t, p_same = 0.0, True
    for _ in range(20):
        za, zb = rng.normal(0, 10, (2, len(NORMAL)))
        ts_z = swap_ts_z(za[None], zb[None])[0]
        t, p, ts = swap_test((truth + za / 1000) / SCALE, (truth + zb / 1000) / SCALE, truth)
        worst_t = max(worst_t, float(np.abs(ts - ts_z).max()))
        p_same &= bool(p == float(np.mean(ts_z <= ts_z[0] + 1e-12)))
    checks.append(dict(항목="d_모의검정=swap_test_패턴별T_max_um(20회)", 값=worst_t, 기준="<1e-6, p 전부 일치",
                       통과=bool(worst_t < 1e-6 and p_same)))

    RESULTS.mkdir(exist_ok=True)
    ck = pd.DataFrame(checks)
    ck.to_csv(RESULTS / "E14_0_checks.csv", index=False, encoding="utf-8-sig")
    pw = power_sim()
    pw.to_csv(RESULTS / "E14_0_power.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(ck.to_string(index=False))
    print(pw.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[0])
    a = ap.parse_args()
    {0: step0}[a.step]()
