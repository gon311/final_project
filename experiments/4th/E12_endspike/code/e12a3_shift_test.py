"""E12-A3: E12-A2의 비교 통계량 불일치(Codex 리뷰)를 고친 위치 이동 검정 + 보정 + 검출력.
규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 2-3절(실행 전 고정)."""
import numpy as np
import pandas as pd
from paths import RESULTS
from e12_edges import W, CROP_DX, POS, ROWS, main, window_edges_track, load

MARGIN, EXCL, ALPHA, NMIN = 10, 12, 0.05, 50
DIPS, STEP_CAL, STEP_INJ = (0.5, 1.0, 1.5, 2.0), 5, 10


def prof_T(D):
    """D (n_img, 2W) -> T(x): 3열 평균 s의 7열 최솟값. 15열 지지구간이 모든 사진에서 유효하지 않으면 NaN."""
    D = np.asarray(D, float)
    ok = ~np.isnan(D).any(0)
    d = np.full(D.shape[1], np.nan); d[ok] = np.median(D[:, ok], 0)
    dev = d - pd.Series(d).rolling(13, center=True, min_periods=5).median().values
    s = pd.Series(dev).rolling(3, center=True).mean()
    sup = pd.Series(ok.astype(float)).rolling(15, center=True, min_periods=15).min().values == 1
    s = s.where(sup)
    return s.rolling(7, center=True, min_periods=7).min().values


def shift_p(T, xs, c, xtrue=None):
    """위치 c의 T를, c에서 ±MARGIN 밖(그리고 진짜 목표 xtrue ±EXCL 밖)의 모든 위치 T와 비교한 순위 p."""
    i = int(np.flatnonzero(xs == c)[0]); t = T[i]
    if np.isnan(t): return dict(t=np.nan, p=np.nan, n=0, present=False, nl=0, nr=0, ll=0, lr=0)
    m = ~np.isnan(T) & (np.abs(xs - c) > MARGIN)
    if xtrue is not None: m &= np.abs(xs - xtrue) > EXCL
    n = int(m.sum()); le = m & (T <= t)
    p = (1 + int(le.sum())) / (1 + n)
    return dict(t=float(t), p=p, n=n, present=bool(n >= NMIN and p <= ALPHA and t < 0),
                nl=int((m & (xs < c)).sum()), nr=int((m & (xs > c)).sum()), ll=int((le & (xs < c)).sum()), lr=int((le & (xs > c)).sum()))


def inject(D, xs, c, a):
    D = np.array(D, float); j = int(np.flatnonzero(xs == c)[0])
    D[:, j - 1:j + 2] += np.array([-a / 2, -a, -a / 2]); return D


def calibrate_and_power(D, xs, xc, power=False):
    """창 안의 진짜 목표 밖 가짜 목표 위치에서 (보정) 오검출률, (검출력) 알려진 스파이크 주입 후 검출률."""
    T = prof_T(D); cal = []
    for c in xs[::STEP_CAL]:
        if abs(c - xc) <= EXCL or np.isnan(T[c - xs[0]]): continue
        r = shift_p(T, xs, int(c), xc); cal.append(r["present"])
    pw = []
    if power:
        for a in DIPS:
            hit = []
            for c in xs[::STEP_INJ]:
                if abs(c - xc) <= EXCL or c - xs[0] < 12 or c - xs[0] > len(xs) - 13: continue
                Ti = prof_T(inject(D, xs, int(c), a))
                hit.append(shift_p(Ti, xs, int(c), xc)["present"])
            pw.append((a, len(hit), int(sum(hit))))
    return cal, pw


def verdict(summ, cal_fpr, power):
    pos = summ[summ.specimen.isin(POS)]
    ctl = int(pos.present.sum())
    g = lambda xc: summ[(summ.specimen == "NO1") & (summ.window_center == xc)].iloc[0]
    a, b = g(8265 - CROP_DX), g(8265)
    pw = lambda xc: power[(power.window_center == xc) & (power.a == 1.0)].power.iloc[0]
    why = []
    if ctl < 4: why.append(f"양성 대조 {ctl}/5")
    if cal_fpr > 0.10: why.append(f"보정 오검출률 {cal_fpr:.2f} > 0.10")
    if min(a.n_null, b.n_null) < NMIN: why.append("오프타겟 위치 수 부족")
    if why: return "판단 불가 (검정 조건 미충족: " + ", ".join(why) + ")"
    pa, pb = bool(a.present), bool(b.present)
    if pa and not pb and pw(8265) >= 0.8: return "5857 후보 검출·8265 미검출 — 센서·렌즈·장면 쪽 후보, 원인 판단 보류"
    if pb and not pa and pw(8265 - CROP_DX) >= 0.8: return "8265 후보 검출·5857 미검출 — 출력 좌표 쪽 후보, 원인 판단 보류"
    return "판단 불가"


if __name__ == "__main__":
    wins = {}  # (specimen, xc) -> D
    for s in POS:
        wins[(s, 8265)] = [e for r in main[main.specimen == s].itertuples() for e in [window_edges_track(load(r), 8265, ROWS["zoom1"])] if e is not None]
    for xc in (8265 - CROP_DX, 8265): wins[("NO1", xc)] = []
    for r in main[main.specimen == "NO1"].itertuples():
        g = load(r)
        for xc in (8265 - CROP_DX, 8265):
            e = window_edges_track(g, xc, ROWS["NO1"])
            if e is not None: wins[("NO1", xc)].append(e)
    rows, cal_rows, pw_rows = [], [], []
    for (s, xc), D in wins.items():
        xs = np.arange(xc - W, xc + W); T = prof_T(D)
        r = shift_p(T, xs, xc)
        rows.append(dict(specimen=s, window_center=xc, n_img=len(D), n_valid_cols=int((~np.isnan(T)).sum()), target_s_px=round(r["t"], 2),
                         n_null=r["n"], p_shift=round(r["p"], 4), present=r["present"], n_null_left=r["nl"], n_null_right=r["nr"],
                         le_target_left=r["ll"], le_target_right=r["lr"]))
        cal, pw = calibrate_and_power(D, xs, xc, power=(s == "NO1"))
        cal_rows.append(dict(specimen=s, window_center=xc, n_pseudo=len(cal), n_false_pos=int(sum(cal))))
        pw_rows += [dict(specimen=s, window_center=xc, a=a, n_pos=n, detected=k, power=round(k / n, 3)) for a, n, k in pw]
    S, C, PW = pd.DataFrame(rows), pd.DataFrame(cal_rows), pd.DataFrame(pw_rows)
    fpr = C.n_false_pos.sum() / C.n_pseudo.sum()
    v = verdict(S, fpr, PW)
    S.to_csv(RESULTS / "E12A3_summary.csv", index=False); C.to_csv(RESULTS / "E12A3_calibration.csv", index=False)
    PW.to_csv(RESULTS / "E12A3_power.csv", index=False)
    pd.DataFrame([dict(calibration_fpr=round(fpr, 3), verdict=v)]).to_csv(RESULTS / "E12A3_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(S.to_string(index=False)); print(C.to_string(index=False)); print(PW.to_string(index=False))
    print(f"보정 오검출률 {fpr:.3f}\n판정: {v}")
