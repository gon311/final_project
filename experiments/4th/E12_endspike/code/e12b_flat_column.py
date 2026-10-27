"""E12-B: 엣지 없는 평평한 밝은 띠의 열 프로파일에서 x=8265 열 이상 여부(위치 이동 검정·보정·주입 검출력).
규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 3절(실행 전 고정). shift_p는 A3와 같은 함수."""
import numpy as np
import pandas as pd
from paths import RESULTS, FIGURES, PROFILES as CACHE
from e12_edges import main, load
from e12a3_shift_test import shift_p

X0, X1, A0, A1 = 5900, 11100, 6000, 11000
XC, EXCL, STEP_CAL, STEP_INJ, GS = 8265, 12, 5, 50, (0.5, 1.0, 2.0, 4.0)
N_MIN, FPR_MAX, VALID_MIN, G_REF, POWER_MIN = 500, 0.10, 0.90, 1.0, 0.8
XS = np.arange(X0, X1)
ANA = (XS >= A0) & (XS < A1)


def sp(T, xs, c, xtrue=None):
    """A3의 shift_p에 B의 '있음' 기준(n >= 500, p <= 0.05, t < 0)을 입힌다. A3 코드는 건드리지 않는다."""
    r = shift_p(T, xs, c, xtrue)
    r["present"] = bool(r["n"] >= N_MIN and r["p"] <= 0.05 and r["t"] < 0)
    return r


def valid_mask(M):
    return ~((M >= 250) | (M < 150)).any(0)


def detrend(M, ok):
    """M (n_img, n_x) 열별 평균 -> 장마다 롤링 15열 중앙값 제거. 무효 열은 NaN."""
    Mv = np.where(ok, M, np.nan)
    med = pd.DataFrame(Mv.T).rolling(15, center=True, min_periods=15).median().values.T
    return Mv - med


def T_profile(M, ok):
    P = np.nanmedian(detrend(M, ok), 0)  # NaN만 있는 열은 NaN
    s = pd.Series(P).rolling(3, center=True).mean()
    T = -s.abs().rolling(7, center=True, min_periods=7).max().values
    T[~ANA] = np.nan  # 비교·검정 위치는 분석 구간(6000~11000)만
    return T, P


# 주입 모양(gray level, 중심 열 기준). 부호는 음수만: T가 |s|라 양수는 대칭이다. 장 방향 중앙값·15열 중앙값 제거 때문에
# 모든 장에 공통이고 폭이 약 7열 이하인 결함만 검출 대상이다(넓은 결함·특정 시편에만 있는 구조는 제거되거나 사라진다).
SHAPES = {"w1": [-1.0], "w3": [-0.5, -1.0, -0.5], "w5": [-0.5, -1.0, -1.0, -1.0, -0.5]}


def inj(M, c, g, shape="w3"):
    M = M.copy(); k = np.array(SHAPES[shape]) * g; j = c - X0 - len(k) // 2; M[:, j:j + len(k)] += k; return M


def pseudo_hits(T):
    """가짜 목표 위치에서 '있음'이 나오는 비율. 같은 T 분포 안에서 순위를 다시 세는 것이라 명목 유의수준 부근이 되기 쉽고,
    위치별 배경의 균질성을 검증하는 독립 대조가 아니다(Codex 3차 리뷰). 순위·상관 구조가 심하게 어긋났는지 보는 점검으로만 쓴다."""
    return [sp(T, XS, int(c), XC)["present"] for c in XS[::STEP_CAL]
            if A0 + 10 <= c < A1 - 10 and abs(c - XC) > EXCL and not np.isnan(T[c - X0])]


def power(M, ok):
    out = []
    for shape in SHAPES:
        for g in GS:
            hit = []
            for c in XS[::STEP_INJ]:
                if not (A0 + 20 <= c < A1 - 20) or abs(c - XC) <= EXCL: continue
                Ti, _ = T_profile(inj(M, int(c), g, shape), ok)
                hit.append(sp(Ti, XS, int(c), XC)["present"])
            out.append(dict(shape=shape, g=g, n_pos=len(hit), detected=int(sum(hit)), power=round(sum(hit) / len(hit), 3)))
    return pd.DataFrame(out)


def verdict(pres, fpr, n_null, valid_frac, pw):
    why = []
    if fpr > FPR_MAX: why.append(f"가짜 위치 검출 비율 {fpr:.2f} > {FPR_MAX}")
    if n_null < N_MIN: why.append("비교 위치 수 부족")
    if valid_frac < VALID_MIN: why.append(f"유효 열 {valid_frac:.0%} < 90%")
    if why: return "판단 불가 (검정 조건 미충족: " + ", ".join(why) + ")"
    if pres: return "평평한 영역에서도 8265 열에 이상 — 열 전체에 걸친 이상(센서·렌즈·장면·ISP 열 처리 모두 해당) 후보, 원인 판단 보류"
    if (pw[pw.g == G_REF].power >= POWER_MIN).all():
        return "모든 장에 공통인 폭 1~5열·g=1 이상의 열 결함 미검출 — 열 전체 결함 가설에 불리, 엣지 영역에 국한된 원인은 배제하지 못함, 원인 판단 보류"
    return "판단 불가"


if __name__ == "__main__":
    z = np.load(CACHE, allow_pickle=True)
    tc = {f: int(c) for f, c in zip(z["filename"].astype(str), z["top_center"])}
    rows = [r for r in main.itertuples() if r.specimen != "NO1"]
    miss = [r.filename for r in rows if r.filename not in tc]
    cnt = pd.Series([r.specimen for r in rows]).value_counts()
    # 입력 완전성: 메타의 본시험 NO2~NO12가 110장(시편당 10장)이고 모두 캐시에 있어야 한다. 아니면 다른 표본으로 판정하지 않는다.
    assert not miss and len(rows) == 110 and (cnt == 10).all() and len(cnt) == 11, f"입력 불완전: 누락 {miss}, 시편별 {cnt.to_dict()}"
    M = np.stack([load(r)[tc[r.filename] - 300:tc[r.filename] - 120, X0:X1].mean(0) for r in rows])
    spec = np.array([r.specimen for r in rows])
    ok = valid_mask(M)
    vf = float(ok[ANA].mean())
    T, P = T_profile(M, ok)
    r = sp(T, XS, XC)
    cal = pseudo_hits(T); fpr = float(np.mean(cal)); PW = power(M, ok)
    S = pd.DataFrame([dict(n_img=len(rows), valid_frac=round(vf, 3), saturated_cols=int((M >= 250).any(0).sum()), dark_cols=int((M < 150).any(0).sum()),
                           target_T_graylevel=round(-r["t"], 3), n_null=r["n"], p_shift=round(r["p"], 4), present=r["present"],
                           n_null_left=r["nl"], n_null_right=r["nr"], le_target_left=r["ll"], le_target_right=r["lr"],
                           n_pseudo=len(cal), n_pseudo_hit=int(sum(cal)), pseudo_hit_rate=round(fpr, 3))])
    S.to_csv(RESULTS / "E12B_summary.csv", index=False)
    pd.DataFrame(dict(x=XS, P_graylevel=P, T=T)).to_csv(RESULTS / "E12B_flat_column.csv", index=False)
    pd.DataFrame([dict(n_pseudo=len(cal), n_pseudo_hit=int(sum(cal)))]).to_csv(RESULTS / "E12B_pseudo_hits.csv", index=False)
    PW.to_csv(RESULTS / "E12B_power.csv", index=False)
    by = []
    for s in sorted(set(spec), key=lambda t: int(t[2:])):
        m = spec == s; Ts, _ = T_profile(M[m], ok)
        q = sp(Ts, XS, XC); by.append(dict(specimen=s, n_img=int(m.sum()), target_T_graylevel=round(-q["t"], 3), p_shift=round(q["p"], 4), present=q["present"]))
    pd.DataFrame(by).to_csv(RESULTS / "E12B_by_specimen.csv", index=False)
    v = verdict(r["present"], fpr, r["n"], vf, PW)
    pd.DataFrame([dict(verdict=v)]).to_csv(RESULTS / "E12B_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(S.to_string(index=False)); print(PW.to_string(index=False)); print(pd.DataFrame(by).to_string(index=False)); print("판정:", v)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    ax[0].plot(XS, P, lw=.5); ax[0].axvline(XC, c="r", ls=":"); ax[0].set_ylabel("P(x) gray level"); ax[0].set_xlim(A0, A1)
    ax[1].plot(XS, P, lw=.8); ax[1].axvline(XC, c="r", ls=":"); ax[1].set_xlim(XC - 60, XC + 60); ax[1].set_xlabel("x (px)"); ax[1].set_ylabel("P(x) zoom")
    fig.tight_layout(); fig.savefig(FIGURES / "E12B_flat_column.png", dpi=110)
