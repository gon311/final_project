"""E14 중앙 구간 대표값 (docs/PREREG_E14.md, docs/EXPERIMENTS_E14.md).

python e14.py --step 0   # 데이터 준비, 재현 확인, 검정 구현 점검, 검정력 모의실험
"""
import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from paths import EDGES, E6_CODE, FIGURES, RES1021, RESULTS, SEED

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


def columns(m):
    """이미지별 열 표: x, Δx(mm), d(µm), 0.1mm 구간 b, 창 소속."""
    rows = []
    for f, g in m.groupby("filename", sort=False):
        _, _, xm, x, d = per_image(g)
        dx = (x - xm) * SCALE
        rows.append(pd.DataFrame(dict(filename=f, specimen=g.specimen.iat[0], mount=int(g["mount"].iat[0]),
                                      x=x, dx_mm=dx, d_um=d * SCALE * 1000, b=np.floor(dx / BIN_MM).astype(int),
                                      W2=np.abs(dx) <= W2_MM, W3=np.abs(dx) <= W3_MM)))
    return pd.concat(rows, ignore_index=True)


def profile(col):
    """시편 프로파일 P(b): 이미지별 구간 평균 → 이미지 간 구간별 중앙값, 모든 이미지에 공통인 구간만."""
    ib = col.groupby(["specimen", "filename", "b"]).d_um.mean().rename("d_b").reset_index()
    nimg = ib.groupby("specimen").filename.nunique()
    cnt = ib.groupby(["specimen", "b"]).filename.nunique()
    keep = cnt[cnt.values == nimg.reindex(cnt.index.get_level_values(0)).values].index
    ib = ib.set_index(["specimen", "b"]).loc[keep].reset_index()
    return ib, ib.groupby(["specimen", "b"]).d_b.median().rename("P")


def rank_corr(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def step1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    col = columns(load())
    ib, P = profile(col)
    specs = sorted(col.specimen.unique(), key=lambda s: int(s[2:]))

    # 프로파일 표 (장착별 포함: mount 0 = 본시험, 1~5 = 재장착 1장씩)
    prof = P.reset_index().assign(mount="all")
    ibm = ib.merge(col[["filename", "mount"]].drop_duplicates(), on="filename")
    pm = ibm.groupby(["specimen", "mount", "b"]).agg(P=("d_b", "median"), n_img=("filename", "nunique")).reset_index()
    prof = pd.concat([prof, pm.assign(mount=pm["mount"].astype(str))], ignore_index=True)
    prof["P_rel_um"] = prof.P - prof.groupby(["specimen", "mount"]).P.transform("median")
    prof["b_center_mm"] = (prof.b + 0.5) * BIN_MM
    prof.to_csv(RESULTS / "E14_profile.csv", index=False, encoding="utf-8-sig")

    # 창 대비 c2, c3 와 끝열 효과 (이미지 단위 → 시편 요약)
    rows = []
    for f, g in col.groupby("filename", sort=False):
        g = g.sort_values("x")
        r1 = g.d_um.mean()
        rows.append(dict(specimen=g.specimen.iat[0], c2=g.d_um[g.W2].mean() - r1, c3=g.d_um[g.W3].mean() - r1,
                         end_L=g.d_um.iloc[:3].mean() - g.d_um.iloc[3:].mean(),
                         end_R=g.d_um.iloc[-3:].mean() - g.d_um.iloc[:-3].mean()))
    ci = pd.DataFrame(rows)
    q = lambda s, k: s.quantile(k)
    con = ci.groupby("specimen").agg(n_img=("c2", "size"),
                                     c2_med=("c2", "median"), c2_q25=("c2", lambda s: q(s, .25)), c2_q75=("c2", lambda s: q(s, .75)),
                                     c3_med=("c3", "median"), c3_q25=("c3", lambda s: q(s, .25)), c3_q75=("c3", lambda s: q(s, .75)),
                                     endL_med=("end_L", "median"), endR_med=("end_R", "median")).loc[specs]
    con.to_csv(RESULTS / "E14_profile_contrast.csv", encoding="utf-8-sig")

    # 잡음/형상 분리
    cb = col.merge(P.reset_index(), on=["specimen", "b"])  # 공통 구간 밖 열은 빠진다
    sn = cb.assign(res=cb.d_um - cb.P).groupby(["specimen", "filename"]).res.std().groupby("specimen").median()
    ns = pd.DataFrame(dict(sigma_noise_um=sn, sigma_shape_um=P.groupby("specimen").std())).loc[specs]
    # 형상 재현 상관 (PREREG 외 기술, Codex 리뷰 3 반영). 파일명 = 촬영 시각 순.
    # 재장착은 '단일 이미지 vs 본시험 중앙값'이므로 같은 꼴의 기준(본시험 단일 이미지 vs 나머지 중앙값)을 둔다.
    # 정합 민감도: ±5구간(±0.5mm) 이동 중 최대 상관을 재장착·기준에 똑같이 적용한다.
    def corr_shift(u, v, k):
        """u를 k구간 밀었을 때 v와의 상관(겹치는 구간만)."""
        uu = u.copy()
        uu.index = uu.index + k
        j = uu.index.intersection(v.index)
        return float(np.corrcoef(uu[j], v[j])[0, 1])

    m0 = ibm[ibm["mount"] == 0]
    rec = []
    for s in specs:
        g0 = m0[m0.specimen == s]
        fs = sorted(g0.filename.unique())
        prof_of = lambda ff: g0[g0.filename.isin(ff)].groupby("b").d_b.median()
        oe = corr_shift(prof_of(fs[0::2]), prof_of(fs[1::2]), 0)
        h = len(fs) // 2
        tb = corr_shift(prof_of(fs[:h]), prof_of(fs[h:]), 0)  # 앞 절반 vs 뒤 절반(시간 블록)
        loo, loo_al = [], []
        for f in fs:
            one, rest = prof_of([f]), prof_of([x for x in fs if x != f])
            loo.append(corr_shift(one, rest, 0))
            loo_al.append(max(corr_shift(one, rest, k) for k in range(-5, 6)))
        rm, rm_al = [], []
        p0 = prof_of(fs)
        for f, h1 in ibm[(ibm.specimen == s) & (ibm["mount"] > 0)].groupby("filename"):
            one = h1.set_index("b").d_b
            rm.append(corr_shift(one, p0, 0))
            rm_al.append(max(corr_shift(one, p0, k) for k in range(-5, 6)))
        rec.append(dict(specimen=s, corr_m0_oddeven=oe, corr_m0_timeblock=tb,
                        corr_m0_loo_med=np.median(loo), corr_m0_loo_aligned_med=np.median(loo_al),
                        corr_remount_med=np.median(rm) if rm else np.nan,
                        corr_remount_aligned_med=np.median(rm_al) if rm_al else np.nan))
    ns = ns.join(pd.DataFrame(rec).set_index("specimen"))
    # R4 편향 진단: 시편 R4(P의 90% 분위수) − 시편 R1(이미지 R1의 중앙값) vs σ_noise, 정상 9시편
    r1s = col.groupby(["specimen", "filename"]).d_um.mean().groupby("specimen").median()
    ns["R4_minus_R1_um"] = P.groupby("specimen").quantile(0.9) - r1s
    rho = rank_corr(ns.loc[NORMAL, "R4_minus_R1_um"], ns.loc[NORMAL, "sigma_noise_um"])
    # 사후 관찰(PREREG 외): 11시편 공통 패턴을 뺀 시편 고유 형상. 공통 패턴은 방식 3의 상수 c가 흡수한다
    Pw = P.unstack("specimen")
    Pc = Pw.sub(Pw.median(), axis=1)  # 시편별 수준 제거
    common_pat = Pc.median(axis=1)
    ns["sigma_specific_um"] = Pc.sub(common_pat, axis=0).std()
    ns["corr_with_common"] = Pc.corrwith(common_pat)
    ns.to_csv(RESULTS / "E14_noise_shape.csv", encoding="utf-8-sig")

    # 사후 관찰(Codex 리뷰 3 반영): 실제 평가에 들어가는 시편 대표값 차이 δ = median(Rk) − median(Rj).
    # 방식 3에서 오차 변화는 (9/8)(δ − mean δ)이므로 |MAE 변화| ≤ mean|(9/8)(δ − mean δ)|. 실측치를 쓰지 않는다.
    wi = col.groupby(["specimen", "filename"]).apply(
        lambda g: pd.Series(dict(R1=g.d_um.mean(), R2=g.d_um[g.W2].mean(), R3=g.d_um[g.W3].mean())))
    rs = wi.groupby("specimen").median().loc[NORMAL]
    wd = []
    for b_, a_ in (("R2", "R1"), ("R3", "R2"), ("R3", "R1")):
        dl = rs[b_] - rs[a_]
        de = 9 / 8 * (dl - dl.mean())
        wd.append(dict(비교=f"{b_}-{a_}", delta_mean_um=dl.mean(), delta_sd_um=dl.std(),
                       MAE변화_상한_um=float(np.abs(de).mean()), 오차변화_max_um=float(np.abs(de).max()),
                       **{f"delta_{s}": dl[s] for s in NORMAL}))
    wd = pd.DataFrame(wd)
    wd.to_csv(RESULTS / "E14_window_delta.csv", index=False, encoding="utf-8-sig")

    # 그림
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhspan(-2.9, 2.9, color="0.88", label="장착 내 σ ±2.9µm")
    for w, ls in ((W2_MM, "--"), (W3_MM, ":")):
        for sg in (-1, 1):
            ax.axvline(sg * w, color="0.4", ls=ls, lw=1, label=(f"W{2 if w == W2_MM else 3} ±{w:.2f}mm" if sg > 0 else None))
    cmap = plt.get_cmap("tab20")
    pa = prof[prof["mount"] == "all"]
    for k, s in enumerate(specs):
        g = pa[pa.specimen == s]
        ax.plot(g.b_center_mm, g.P_rel_um, ls="-" if s in NORMAL else "--", lw=1.4, color=cmap(k), label=s)
    ax.set_xlabel("창 중앙 기준 위치 x − xm (mm)")
    ax.set_ylabel("시편 프로파일 P(b) − 중앙값 (µm)")
    ax.set_title("E14-1 열별 직경 프로파일 (10/21, 시편별 이미지 중앙값, 0.1mm 구간)")
    ax.legend(fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / "E14_profile.png", dpi=110)

    pd.set_option("display.width", 220)
    print(con.round(2).to_string())
    print(ns.round(3).to_string())
    a11 = con.c3_med.abs().median()
    a9 = con.loc[NORMAL, "c3_med"].abs().median()
    print(f"\n|c3| 시편 중앙값의 중앙값: 11시편 {a11:.2f}µm, 정상 9시편 {a9:.2f}µm (기준 2.9µm)")
    print(f"R4−R1 vs σ_noise Spearman (정상 9): {rho:.3f}")
    print(f"[사후] 공통 패턴 SD {common_pat.std():.2f}µm")
    print(wd[["비교", "delta_mean_um", "delta_sd_um", "MAE변화_상한_um", "오차변화_max_um"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[0, 1])
    a = ap.parse_args()
    {0: step0, 1: step1}[a.step]()
