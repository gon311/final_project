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


IMG_CANDS = ["R0", "R1", "R2", "R3", "R2L", "R3L", "R1m", "R2m", "R3m"]
MAIN = [("R1", "R2"), ("R2", "R3")]  # (A 기준, B 후보), PREREG 5.2
DESC = [("R0", "R1"), ("R0", "R2L"), ("R0", "R3L"), ("R0", "R2"), ("R0", "R3"), ("R1", "R3"), ("R1", "R4"), ("R1", "R5")]


def image_candidates(g):
    """PREREG 2.1. 모든 후보가 W1 공통 θ·xm을 쓴다. 단위 px."""
    r0, th, xm, x, d = per_image(g)
    dx = (x - xm) * SCALE
    yt, yb = g.y_top.to_numpy(), g.y_bot.to_numpy()
    out = dict(R0=r0)
    for k, w in (("1", np.inf), ("2", W2_MM), ("3", W3_MM)):
        sel = np.abs(dx) <= w
        out["R" + k], out["R" + k + "m"] = d[sel].mean(), np.median(d[sel])
        if k != "1":  # 창 안에서만 강건 직선적합, 창 중앙에서 평가
            xs = x[sel]
            xc = xs.mean()
            out["R" + k + "L"] = (np.polyval(robust_fit(xs, yb[sel]), xc) - np.polyval(robust_fit(xs, yt[sel]), xc)) * np.cos(th)
    return out


def step2():
    m = load()
    rows = []
    for f, g in m.groupby("filename", sort=False):
        rows.append(dict(filename=f, specimen=g.specimen.iat[0], mount=int(g["mount"].iat[0]), 실측mm=g["실측mm"].iat[0],
                         pin_cross50_px=g.pin_cross50_px.iat[0], **image_candidates(g)))
    img = pd.DataFrame(rows)
    img.to_csv(RESULTS / "E14_by_image.csv", index=False, encoding="utf-8-sig")
    col = columns(m)

    spec_tabs = {}
    for sel in ("all", "burst"):
        im = img if sel == "all" else img[img["mount"] == 0]
        t = im.groupby("specimen").agg(실측mm=("실측mm", "first"), pin_px=("pin_cross50_px", "median"),
                                       **{c: (c, "median") for c in IMG_CANDS})
        t_ratio = im.assign(**{c: im[c] / im.pin_cross50_px for c in IMG_CANDS}).groupby("specimen")[IMG_CANDS].median()
        _, P = profile(col[col.filename.isin(im.filename)])
        t["R4"] = P.groupby("specimen").quantile(0.9) / (SCALE * 1000)  # µm → px (PREREG 2.2)
        t["R5"] = P.groupby("specimen").max() / (SCALE * 1000)
        for c in ("R4", "R5"):
            t_ratio[c] = t[c] / t.pin_px  # 시편 단위 후보는 시편 핀 중앙값으로 나눈다 (사전등록 구현)
        # 사후 민감도(Codex 리뷰 4): 이미지별 d(x)/pin을 먼저 구한 뒤 프로파일 → 분위수·최댓값. 방식 4에만 쓴다
        cn = col[col.filename.isin(im.filename)].merge(im[["filename", "pin_cross50_px"]], on="filename")
        _, Pn = profile(cn.assign(d_um=cn.d_um / (SCALE * 1000) / cn.pin_cross50_px))
        t_ratio["R4n"], t_ratio["R5n"] = Pn.groupby("specimen").quantile(0.9), Pn.groupby("specimen").max()
        spec_tabs[sel] = (t, t_ratio)
    bs = pd.concat({k: v[0].join(v[1].add_prefix("ratio_")) for k, v in spec_tabs.items()}, names=["사진선택"]).reset_index()
    bs.to_csv(RESULTS / "E14_by_specimen.csv", index=False, encoding="utf-8-sig")

    cands = IMG_CANDS + ["R4", "R5", "R4n", "R5n"]
    lrows = []
    for sel, (t, tr) in spec_tabs.items():
        for c in cands:
            for meth, fitter, src in (("3_V1고정", C.fit_fixed_scale, t), ("4_핀비율선형", C.fit_linear, tr)):
                if c not in src:  # R4n·R5n은 비율 정의라 방식 4만
                    continue
                tab = pd.DataFrame({"실측mm": t["실측mm"], "px": src[c]})
                err = loso_err(tab, fitter)
                fn, _ = fitter(tab.loc[NORMAL])
                lrows.append(dict(사진선택=sel, 후보=c, 방식=meth, MAE_um=np.abs(err).mean(), MAX_um=np.abs(err).max(),
                                  **{f"err_{s}": e for s, e in zip(NORMAL, err)},
                                  **{f"pred_{s}": fn(float(tab.loc[s, "px"])) for s in ("NO11", "NO12")}))
    loso = pd.DataFrame(lrows)
    loso.to_csv(RESULTS / "E14_loso.csv", index=False, encoding="utf-8-sig")

    # PREREG 5: 방식 3, all
    t = spec_tabs["all"][0]
    truth = t.loc[NORMAL, "실측mm"].to_numpy()
    mae = lambda c: mae_m3(t.loc[NORMAL, c].to_numpy(), truth)
    trows = []
    for kind, pairs in (("주", MAIN), ("기술", DESC)):
        for a_, b_ in pairs:
            tt, p, _ = swap_test(t.loc[NORMAL, a_].to_numpy(), t.loc[NORMAL, b_].to_numpy(), truth)
            trows.append(dict(구분=kind, 비교=f"{b_} vs {a_}", MAE_A_um=mae(a_), MAE_B_um=mae(b_), T_um=tt,
                              MAE감소_um=-tt, p_단측=p))
    tests = pd.DataFrame(trows)
    # 보정 Holm(PREREG 5.2): 작은 p ≤ 9/512 이면 기각, 그다음 나머지 ≤ 20/512
    mi = tests.index[tests.구분 == "주"]
    order = tests.loc[mi, "p_단측"].sort_values().index
    rej = {i: False for i in mi}
    if tests.at[order[0], "p_단측"] <= CUT_1 + 1e-12:
        rej[order[0]] = True
        rej[order[1]] = tests.at[order[1], "p_단측"] <= CUT_2 + 1e-12
    tests["Holm보정_기각"] = [rej.get(i, "기술") for i in tests.index]
    tests["판정"] = ["기술" if i not in mi else ("개선" if rej[i] and tests.at[i, "MAE감소_um"] >= 1 else "구분 불가")
                   for i in tests.index]
    tests.to_csv(RESULTS / "E14_tests.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print(tests.round(3).to_string(index=False))
    piv = loso.pivot_table(index="후보", columns=["방식", "사진선택"], values="MAE_um").loc[cands]
    print(piv.round(2).to_string())


INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"  # dataviz 기본 범주 순서
DEMO_FILE = "20261021_153554.jpg"  # NO2 본시험 첫 장 (그림 1·2 예시)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["AppleGothic", "DejaVu Sans"], "axes.unicode_minus": False,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def _demo_gray(v):
    """측정(cv2.imread)과 같은 좌표계: EXIF 회전(이 사진들은 orientation 3 = 180°)을 적용한 회색조."""
    from PIL import Image, ImageOps
    from paths import RAW_1021
    Image.MAX_IMAGE_PIXELS = None  # 16320x7532 원본, 신뢰하는 내부 자료
    return ImageOps.exif_transpose(Image.open(RAW_1021 / v.folder / DEMO_FILE)).convert("L")


def fig_overlay(m):
    """그림 1: 실제 사진 위 측정 구간(베어링 구간, W1·W2·W3, 31개소)과 열별 엣지 확대."""
    plt = _plt()
    g = m[m.filename == DEMO_FILE].sort_values("x")
    v = pd.read_csv(RES1021 / "values_by_image_1021.csv", encoding="utf-8-sig").set_index("filename").loc[DEMO_FILE]
    _, th, xm, x, _ = per_image(g)
    L = (x.max() - x.min()) / 0.6  # measure_section은 베어링 구간 양끝 20%를 뺀다 → W1 = 0.6·구간
    sec = (xm - L / 2, xm + L / 2)
    px = lambda mm: mm / SCALE
    x0, x1 = int(xm - px(6.5)), int(xm + px(6.5))
    y0, y1 = int(v.bearing_top_y - 140), int(v.bearing_bot_y + 140)
    full = _demo_gray(v)
    im = np.asarray(full.crop((x0, y0, x1, y1)))

    fig = plt.figure(figsize=(12, 7.6))
    ph = 0.60  # 사진 축 높이(그림 비율). 가로는 실제 종횡비를 지키도록 계산
    pw = ph * 7.6 * (x1 - x0) / (y1 - y0) / 12
    ax = fig.add_axes([0.05, 0.33, pw, ph])
    ax.imshow(im, cmap="gray", extent=(x0, x1, y1, y0), vmin=0, vmax=255)
    bands = [(x.min(), x.max(), S1, f"W1 측정 창 {(x.max() - x.min()) * SCALE:.1f}mm — 111열 (R0·R1)"),
             (xm - px(W2_MM), xm + px(W2_MM), S2, f"W2 ±{W2_MM:.2f}mm — 중앙 10개소 상당 (R2)"),
             (xm - px(W3_MM), xm + px(W3_MM), S3, f"W3 ±{W3_MM:.2f}mm — 중앙 5개소 상당 (R3)")]
    for a, b, c, _ in bands:
        ax.axvspan(a, b, color=c, alpha=0.22, lw=0)
    # 31개소(멘토 설명: 10.2mm에 고르게)는 사진을 가리지 않게 축 바깥 배경 쪽에만 짧게
    x31 = xm + px(np.linspace(-5.1, 5.1, 31))
    ax.vlines(x31, y0, v.bearing_top_y - 8, color=S4, lw=1.1)
    ax.vlines(x31, v.bearing_bot_y + 8, y1, color=S4, lw=1.1)
    for xs_ in sec:
        ax.axvline(xs_, color="white", lw=1.3, ls="--")
    zx, zy = (xm - px(0.5), xm + px(0.5)), (v.bearing_top_y - 25, v.bearing_top_y + 25)
    ax.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="white", lw=1.5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_yticks([])
    ax.set_title(f"측정 구간 — NO2 실제 사진\n(10/21, {DEMO_FILE}, 실제 비율)", loc="left", color=INK, fontsize=10)

    # 사진 아래 구간 막대: 각 창의 폭을 mm 눈금 위에 나란히 (사진과 같은 x축)
    ab = fig.add_axes([0.05, 0.07, pw, 0.24], sharex=ax)
    rows = [(sec[0], sec[1], "white", f"베어링 구간 (검출 ≈{L * SCALE:.1f}mm, 흰 점선)")] + bands
    for i, (a, b, c, lab) in enumerate(rows):
        yy = len(rows) - i
        ab.plot([a, b], [yy, yy], color=c if c != "white" else INK2, lw=6, solid_capstyle="butt")
        ab.text(x1 + px(0.15), yy, lab, va="center", fontsize=8.5, color=INK)
    ab.plot(x31, [0] * 31, "|", ms=9, mew=1.3, color=S4)
    ab.text(x1 + px(0.15), 0, "31개소 — 10.2mm 균등 (멘토 설명, 노란 선)", va="center", fontsize=8.5, color=INK)
    ab.set_ylim(-0.8, len(rows) + 0.8)
    ab.set_yticks([])
    ab.spines["left"].set_visible(False)
    ticks = np.arange(-6, 6.1, 2)
    ab.set_xticks(xm + px(ticks), [f"{t:+.0f}" if t else "0" for t in ticks])
    ab.set_xlabel("창 중앙 기준 위치 (mm)")

    # 확대: 윗엣지 ±0.5mm, 열마다 cross50 엣지 점과 W1 강건 직선
    az = fig.add_axes([0.52, 0.45, 0.45, 0.45])
    zi = np.asarray(full.crop((int(zx[0]), int(zy[0]), int(zx[1]), int(zy[1]))))
    az.imshow(zi, cmap="gray", extent=(int(zx[0]), int(zx[1]), int(zy[1]), int(zy[0])), vmin=0, vmax=255, aspect="auto")
    sel = (g.x >= zx[0]) & (g.x <= zx[1])
    for xc in g.x[sel]:
        az.axvline(xc, color=S1, lw=0.4, alpha=0.6)
    az.plot(g.x[sel], g.y_top[sel], "o", ms=3.5, color=S4, mec="#000", mew=0.4, label="열별 엣지 (cross50, 서브픽셀)")
    pt = robust_fit(x, g.y_top.to_numpy())
    xx = np.array(zx)
    az.plot(xx, np.polyval(pt, xx), color=S2, lw=2, label="W1 강건 직선적합 (R0의 윗변)")
    az.set_xlim(*zx)
    az.set_ylim(zy[1], zy[0])
    az.set_xticks([])
    az.set_yticks([])
    az.set_title("확대(흰 사각형): 윗엣지 ±0.5mm, 2px 간격 열", loc="left", fontsize=9, color=INK)
    az.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=8, frameon=False)
    fig.text(0.52, 0.30, f"스케일 {SCALE * 1000:.1f}µm/px · 직경 ≈ {v.bearing_cross50_px:.0f}px ≈ 20mm\n"
                         "열 직경 d(x) = (아랫엣지 − 윗엣지) · cos θ\n"
                         "R0 = 위·아래 직선적합의 창 중앙 거리 · R1~R3 = 창 안 d(x)의 평균",
             fontsize=8.5, color=INK2)
    fig.savefig(FIGURES / "E14_fig1_overlay.png", dpi=110)
    plt.close(fig)


def fig_representatives(m):
    """그림 2: 대표값 만드는 법. (a) 이미지 1장의 d(x)와 R0~R3, (b) 시편 프로파일과 R4·R5."""
    plt = _plt()
    g = m[m.filename == DEMO_FILE]
    r = image_candidates(g)
    _, _, xm, x, d = per_image(g)
    dx = (x - xm) * SCALE
    base = r["R0"]
    um = lambda p: (p - base) * SCALE * 1000
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))
    for ax in (a1, a2):
        ax.axvspan(-W2_MM, W2_MM, color=S2, alpha=0.08, lw=0)
        ax.axvspan(-W3_MM, W3_MM, color=S3, alpha=0.14, lw=0)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xlabel("창 중앙 기준 위치 (mm)")
    a1.plot(dx, um(d), "o", ms=3, color="#8a8984", label="열 직경 d(x) (111열)")
    span = {"R0": (dx.min(), dx.max()), "R1": (dx.min(), dx.max()), "R2": (-W2_MM, W2_MM), "R3": (-W3_MM, W3_MM)}
    lab = {"R0": "R0 직선적합 (현재)", "R1": "R1 전체 평균", "R2": "R2 중앙 10개소 평균", "R3": "R3 중앙 5개소 평균"}
    for k, c, ls, z in (("R1", S1, "-", 3), ("R2", S2, "-", 3), ("R3", S3, "-", 3), ("R0", INK, (0, (4, 3)), 4)):
        a1.plot(span[k], [um(r[k])] * 2, color=c, lw=2.2, ls=ls, zorder=z, label=f"{lab[k]} {um(r[k]):+.1f}µm")
    a1.set_ylabel("R0 대비 (µm)")
    a1.set_ylim(-62, 78)
    a1.set_title(f"(a) 이미지 1장 — NO2 {DEMO_FILE}", loc="left", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)

    col = columns(m[m.specimen == "NO2"])
    _, P = profile(col)
    P = P.loc["NO2"]
    r0s = img_r0 = np.median([per_image(h)[0] for _, h in m[m.specimen == "NO2"].groupby("filename")])
    pu = P.to_numpy() - r0s * SCALE * 1000
    bc = (P.index.to_numpy() + 0.5) * BIN_MM
    a2.plot(bc, pu, "-o", ms=3, lw=1, color="#8a8984", label="시편 프로파일 P(b) (10장 중앙값, 0.1mm 구간)")
    for q, c, nm in ((np.quantile(pu, 0.9), S4, "R4 90% 분위수 (탐색)"), (pu.max(), S5, "R5 최댓값 (탐색)"),
                     (np.median(pu), INK, "참고: P의 중앙값")):
        a2.axhline(q, color=c, lw=2 if c != INK else 1, ls="-" if c != INK else ":", label=f"{nm} {q:+.1f}µm")
    a2.set_title("(b) 시편 단위 — NO2 프로파일과 마이크로미터식 후보", loc="left", fontsize=10)
    a2.legend(fontsize=8, frameon=False, loc="upper center")
    a2.annotate("양끝 열 상승\n(11시편 공통)", xy=(-2.8, pu[0]), xytext=(-2.2, pu[0] + 5), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.text(0.5, 0.005, "주황 음영 = W2(±1.53mm), 초록 음영 = W3(±0.68mm). 같은 θ·창 중앙을 모든 후보가 공유한다.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "E14_fig2_representatives.png", dpi=110)
    plt.close(fig)


def fig_results():
    """그림 4: MAE 비교 + 시편별 |오차| 기울기. 그림 5: 실측 vs 예측(공차 띠, NG)."""
    plt = _plt()
    lo = pd.read_csv(RESULTS / "E14_loso.csv")
    te = pd.read_csv(RESULTS / "E14_tests.csv").set_index("비교")
    bs = pd.read_csv(RESULTS / "E14_by_specimen.csv")
    a = lo[(lo.사진선택 == "all") & (lo.방식 == "3_V1고정")].set_index("후보")
    order = ["R0", "R1", "R2", "R3"]
    names = {"R0": "R0\n직선적합(현재)", "R1": "R1\n전체 평균", "R2": "R2\n중앙 10개소", "R3": "R3\n중앙 5개소"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    xs = np.arange(len(order))
    a1.axhline(E6_MAE["0_평균찍기"], color=INK2, ls="--", lw=1)
    a1.text(1.5, E6_MAE["0_평균찍기"] - 0.2, f"참고: 평균찍기 {E6_MAE['0_평균찍기']:.2f}µm", ha="center",
            va="top", fontsize=8.5, color=INK2)
    cols = [INK2, S1, S2, S3]
    for i, k in enumerate(order):
        a1.plot([i, i], [0, a.loc[k, "MAE_um"]], color=cols[i], lw=3, solid_capstyle="round")
        a1.plot(i, a.loc[k, "MAE_um"], "o", ms=9, color=cols[i])
        a1.text(i, a.loc[k, "MAE_um"] + 0.25, f"{a.loc[k, 'MAE_um']:.2f}", ha="center", fontsize=9, color=INK)
    for (i, j, key) in ((1, 2, "R2 vs R1"), (2, 3, "R3 vs R2")):
        yb = max(a.loc[order[i], "MAE_um"], a.loc[order[j], "MAE_um"]) + 0.9
        a1.annotate("", xy=(j, yb), xytext=(i, yb), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
        a1.text((i + j) / 2, yb + 0.12, f"p={te.loc[key, 'p_단측']:.2f}\n구분 불가", ha="center", fontsize=8, color=INK)
    a1.set_xticks(xs, [names[k] for k in order], fontsize=8.5)
    a1.set_ylim(0, 11.5)
    a1.set_ylabel("정상 9시편 LOSO MAE (µm)")
    a1.set_title("(a) 창을 좁히면 MAE가 줄었나 — 방식 3, all", loc="left", fontsize=10)

    errs = np.abs(a.loc[["R1", "R2", "R3"], [f"err_{s}" for s in NORMAL]].to_numpy())  # 3 x 9
    ly = {}  # 라벨이 겹치지 않게 최소 간격 0.9µm로 위로 민다
    for j in np.argsort(errs[2]):
        y = errs[2, j]
        if ly:
            y = max(y, max(ly.values()) + 0.9)
        ly[j] = y
    for j, s in enumerate(NORMAL):
        a2.plot([0, 1, 2], errs[:, j], "-o", ms=4, lw=1, color="#a8a7a2")
        a2.text(2.06, ly[j], s, fontsize=7.5, va="center", color=INK2)
    a2.plot([0, 1, 2], errs.mean(1), "-o", ms=7, lw=2.5, color=S1, label="평균(=MAE)")
    a2.set_xticks([0, 1, 2], ["R1 전체", "R2 중앙 10", "R3 중앙 5"])
    a2.set_xlim(-0.2, 2.35)
    a2.set_ylabel("시편별 |오차| (µm)")
    a2.set_title("(b) 시편별로 보면 — 일관된 감소가 없다", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "E14_fig4_mae.png", dpi=110)
    plt.close(fig)

    truth = bs[bs.사진선택 == "all"].set_index("specimen")["실측mm"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.2), sharex=True, sharey=True)
    lim = (19.94, 20.06)
    for ax, k, c in zip(axs, ["R1", "R2", "R3"], [S1, S2, S3]):
        ax.axhspan(C.TOL_LO, C.TOL_HI, color=S3, alpha=0.10, lw=0)
        ax.axvspan(C.TOL_LO, C.TOL_HI, color=GRID, alpha=0.35, lw=0)
        ax.plot(lim, lim, color=INK2, lw=0.8, ls=":")
        pr = {s: truth[s] + a.loc[k, f"err_{s}"] / 1000 for s in NORMAL}
        ax.plot([truth[s] for s in NORMAL], [pr[s] for s in NORMAL], "o", ms=7, color=c, mec="white", mew=1,
                label="정상 9 (LOSO 예측)")
        for s, mk in (("NO11", "v"), ("NO12", "^")):
            p = a.loc[k, f"pred_{s}"]
            ok = C.judge(p) == "OK"
            ax.plot(truth[s], p, mk, ms=10, color="white" if ok else INK, mec=INK, mew=1.5,
                    label=None)
            ax.annotate(f"{s} {'OK → 놓침' if ok else 'NG 검출'}", (truth[s], p),
                        xytext=(0, 13) if s == "NO12" else (8, 8),
                        textcoords="offset points", ha="center" if s == "NO12" else "left", fontsize=8,
                        color=S2 if ok else INK, fontweight="bold" if ok else "normal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{k}  MAE {a.loc[k, 'MAE_um']:.2f}µm", loc="left", fontsize=10)
        ax.set_xlabel("마이크로미터 실측 (mm)")
    axs[0].set_ylabel("이미지 예측 (mm, 방식 3)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.5, 0.005, "초록 띠 = 예측이 공차(20.010~20.030mm) 안 → OK 판정. 회색 세로 띠 = 실측이 공차 안. "
                         "NG 2개(▼NO11, ▲NO12)는 정상 9시편 전체로 맞춘 식의 예측. 흰 기호 = 불량을 OK로 놓친 경우.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "E14_fig5_pred_vs_true.png", dpi=110)
    plt.close(fig)


def fig_profile():
    """그림 6: 열별 프로파일 — 시편별(회색)과 11시편 공통 패턴(굵은 선)."""
    plt = _plt()
    pr = pd.read_csv(RESULTS / "E14_profile.csv")
    pa = pr[pr["mount"] == "all"]
    w = pa.pivot(index="b_center_mm", columns="specimen", values="P_rel_um")
    common_pat = w.median(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhspan(-2.9, 2.9, color=GRID, alpha=0.6, lw=0, label="장착 내 σ ±2.9µm")
    ax.axvspan(-W2_MM, W2_MM, color=S2, alpha=0.07, lw=0, label="W2 (중앙 10개소)")
    ax.axvspan(-W3_MM, W3_MM, color=S3, alpha=0.12, lw=0, label="W3 (중앙 5개소)")
    for k, s in enumerate(w.columns):
        ax.plot(w.index, w[s], color="#b5b4af", lw=0.9, ls="-" if s in NORMAL else "--",
                label=("시편 11개 (정상 실선, NG 점선)" if k == 0 else None))
    ax.plot(common_pat.index, common_pat, color=S1, lw=2.6, label="11시편 공통 패턴 (구간별 중앙값)")
    for xa_, txt in ((-2.8, "왼쪽 끝 상승"), (2.75, "오른쪽 끝 상승")):
        ax.annotate(txt, xy=(xa_, common_pat.loc[common_pat.index[np.argmin(np.abs(common_pat.index - xa_))]]),
                    xytext=(xa_ * 0.72, 75), fontsize=8.5, color=INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("창 중앙 기준 위치 (mm)")
    ax.set_ylabel("시편 프로파일 − 시편 중앙값 (µm)")
    ax.set_title("열별 직경 프로파일 — 시편이 달라도 모양이 같다 (사후 관찰: 촬영계 고정 패턴 추정)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)
    ax.set_ylim(-65, 115)
    fig.tight_layout()
    fig.savefig(FIGURES / "E14_profile.png", dpi=110)
    plt.close(fig)


def fig_pipeline():
    """그림 3: 측정 파이프라인에서 E14가 바꾼 단계(대표값)만 강조한 흐름도."""
    from matplotlib.patches import FancyBboxPatch
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    W, H = 2.15, 1.05

    def box(x, y, title, body, hi=False):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eaf2fc" if hi else "#f4f4f2", ec=S1 if hi else "#b5b4af", lw=2 if hi else 1))
        ax.text(x + W / 2, y + H - 0.22, title, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(x + W / 2, y + 0.38, body, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))

    top, bot = 3.15, 0.75
    xs = [0.15, 3.1, 6.05, 9.0]
    row1 = [("① 사진", "16320×7532 백라이트\n실루엣 (10/21)"),
            ("② 베어링 구간 검출", "약 9.5mm, 양끝 20% 제외\n→ 측정 창 W1 ≈5.7mm"),
            ("③ 열별 엣지", "2px 간격 111열\n위·아래 cross50 서브픽셀"),
            ("④ 열 직경 d(x)", "(아래 − 위) · cos θ\nθ = W1 직선적합 기울기")]
    row2 = [("⑤ 이미지 대표값 (px)", "R0 직선적합 (현재)\nR1 전체 · R2 중앙10 · R3 중앙5", True),
            ("⑥ 시편 대표값", "이미지 10~15장의 중앙값\n(R4·R5는 시편 프로파일)", False),
            ("⑦ 보정 → mm", "방식 3: mm = 0.026092·px + c\nc는 LOSO로 8시편에서 적합", False),
            ("⑧ 판정", "공차 20.010~20.030mm\nOK / NG", False)]
    for x, (t, b) in zip(xs, row1):
        box(x, top, t, b)
    for x, (t, b, hi) in zip(xs, row2):
        box(x, bot, t, b, hi)
    for i in range(3):
        arrow(xs[i] + W, top + H / 2, xs[i + 1], top + H / 2)
        arrow(xs[i] + W, bot + H / 2, xs[i + 1], bot + H / 2)
    ym = (top + bot + H) / 2  # ④ 아래에서 꺾어 ⑤ 위로
    ax.plot([xs[3] + W / 2, xs[3] + W / 2, xs[0] + W / 2], [top, ym, ym], color=INK2, lw=1.1)
    arrow(xs[0] + W / 2, ym, xs[0] + W / 2, bot + H)
    ax.text(xs[0] + W / 2, bot - 0.32, "E14가 바꾼 단계는 ⑤뿐 — 나머지는 E6(10/21)와 동일",
            ha="center", fontsize=9, color=S1, fontweight="bold")
    ax.text(xs[2] + W / 2, bot - 0.32, "평가: 정상 9시편 LOSO MAE + 시편 내 교환 무작위화 검정",
            ha="center", fontsize=8.5, color=INK2)
    ax.set_title("측정 흐름과 E14의 비교 지점", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "E14_fig3_pipeline.png", dpi=110)
    plt.close(fig)


def fig_power():
    """그림 7: 개별 비교의 보정 임계값 통과율(검정력 근사)과 이번 관측값의 위치."""
    plt = _plt()
    pw = pd.read_csv(RESULTS / "E14_0_power.csv")
    te = pd.read_csv(RESULTS / "E14_tests.csv").set_index("비교")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, col, ttl in ((a1, "기각률_보정9of512", "1단계 기준 p ≤ 9/512"), (a2, "기각률_보정20of512", "2단계 기준 p ≤ 20/512")):
        ax.axhline(0.05 if "20" in col else 0.025, color=INK2, ls=":", lw=1)
        for rho, c in ((0.5, S1), (0.8, S2), (0.95, S3)):
            g = pw[pw.rho == rho]
            ax.plot(g.MAE감소_um, g[col], "-o", ms=7, lw=2, color=c, label=f"ρ = {rho}")
            ax.text(5.12, g[col].iloc[-1], f"ρ={rho}", fontsize=8, va="center", color=INK2)
        ax.axvspan(-1.2, 0, color=GRID, alpha=0.5, lw=0)
        obs = [(k, te.loc[k, "MAE감소_um"]) for k in ("R2 vs R1", "R3 vs R2")]
        for (key, d), mk in zip(obs, ("v", "^")):
            ax.plot(d, 0.02, mk, ms=9, color=INK)
        # 관측 MAE 변화는 '실제 효과'의 추정치일 뿐 같은 양이 아니라 참고 표시
        ax.text(-1.1, 0.30, "참고: 이번 관측\n" + "\n".join(f"{'▼' if i == 0 else '▲'} {k} {d:+.2f}µm" for i, (k, d) in enumerate(obs))
                + "\n(MAE가 오히려 늘어남)", fontsize=7.5, color=INK, va="top")
        ax.axvline(1, color=S4, lw=1.2, ls="--")
        ax.text(1.05, 0.93, "판정 하한\n1µm 감소", fontsize=7.5, color=INK2, va="top")
        ax.set_xlim(-1.2, 5.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("실제 MAE 감소 (µm)  ← 늘어남 | 줄어듦 →")
        ax.set_title(ttl, loc="left", fontsize=10)
    a1.set_ylabel("임계값 통과율 (2000회 모의)")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left", title="두 대표값의 상관", title_fontsize=8)
    fig.text(0.5, 0.005, "정규 생성 모형·n=9에서 비교 하나를 따로 모의한 값이다. Holm 2단계와 '1µm 이상 감소' 조건을 함께 넣은 "
                         "실제 판정의 검정력은 이보다 낮거나 같다. 점선 가로선 = 명목 α.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "E14_fig7_power.png", dpi=110)
    plt.close(fig)


def step3():
    """보고서 그림 (측정·판정에 쓰지 않음). 그림 1만 원본 사진을 읽는다(PREREG 부록 B, 시각화 예외)."""
    FIGURES.mkdir(exist_ok=True)
    m = load()
    fig_overlay(m)
    fig_representatives(m)
    fig_pipeline()
    fig_results()
    fig_profile()
    fig_power()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[0, 1, 2, 3])
    a = ap.parse_args()
    {0: step0, 1: step1, 2: step2, 3: step3}[a.step]()
