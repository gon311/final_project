"""E12-A2: E12-A의 폭·최저점 규칙을 순위 검정으로 바꿔 NO1 크롭 프레임을 다시 본다.
규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 2-2절(실행 전 고정)."""
import numpy as np
import pandas as pd
from PIL import Image
from paths import RESULTS, UM_PER_PX
from e12_edges import TOL, W, CROP_DX, POS, ROWS, main, window_edges, load

MARGIN, EDGE, ALPHA = 8, 10, 0.05  # 목표 ±8열은 오프타겟에서 제외, 창 양끝 10열 제외(롤링 중앙값 가장자리 효과)


def stat3(dev):
    """3열 평균 편차 s(x). 좁은 홈의 깊이를 열 하나의 잡음보다 안정적으로 본다."""
    return pd.Series(dev).rolling(3, center=True).mean().values


def rank_test(D, xs, xc):
    d = np.nanmedian(D, 0)
    dev = d - pd.Series(d).rolling(13, center=True, min_periods=5).median().values
    s = stat3(dev)
    on = np.abs(xs - xc) <= TOL
    off = (np.abs(xs - xc) > MARGIN) & (np.arange(len(xs)) >= EDGE) & (np.arange(len(xs)) < len(xs) - EDGE) & ~np.isnan(s)
    t = np.nanmin(s[on])
    n_off = int(off.sum())
    p = (1 + int((s[off] <= t).sum())) / (1 + n_off)
    return dict(target_x=int(xs[on][int(np.nanargmin(s[on]))]), target_s_px=round(float(t), 2), target_s_um=round(float(t) * UM_PER_PX),
                n_off=n_off, p_rank=round(p, 4), present=bool(p <= ALPHA and t < 0)), d


def run(s, kind, xc, imgs):
    D = [e for r in imgs for e in [window_edges(load(r), xc, ROWS[kind])] if e is not None]
    xs = np.arange(xc - W, xc + W)
    res, d = rank_test(D, xs, xc)
    return dict(specimen=s, window_center=xc, n_img=len(D), **res), d


rows, prof = [], []
for s in POS:
    r, d = run(s, "zoom1", 8265, list(main[main.specimen == s].itertuples())); rows.append(r)
    prof += [dict(specimen=s, window_center=8265, x=int(x), dev_median_px=float(v)) for x, v in zip(np.arange(8265 - W, 8265 + W), d)]
no1 = list(main[main.specimen == "NO1"].itertuples())
for xc in (8265 - CROP_DX, 8265):
    r, d = run("NO1", "NO1", xc, no1); rows.append(r)
    prof += [dict(specimen="NO1", window_center=xc, x=int(x), dev_median_px=float(v)) for x, v in zip(np.arange(xc - W, xc + W), d)]
S = pd.DataFrame(rows)
S.to_csv(RESULTS / "E12A2_summary.csv", index=False)
pd.DataFrame(prof).to_csv(RESULTS / "E12A2_profiles.csv", index=False)
print(S.to_string(index=False))

pos = S[S.specimen.isin(POS)]
repro = int(pos.present.sum())
f5857 = bool(S[(S.specimen == "NO1") & (S.window_center == 8265 - CROP_DX)].present.iloc[0])
f8265 = bool(S[(S.specimen == "NO1") & (S.window_center == 8265)].present.iloc[0])
if repro < 4: v = "양성 대조 미재현 — 방법 문제, 중단"
elif f5857 and not f8265: v = "센서·렌즈·장면 쪽 (출력 좌표 고정 아님)"
elif f8265 and not f5857: v = "출력 좌표에 붙은 원인 (JPEG/ISP 후처리 쪽)"
else: v = "판단 불가"
print(f"양성 대조 {repro}/5, NO1 5857 {f5857}, 8265 {f8265} -> {v}")
pd.DataFrame([dict(positive_control=f"{repro}/5", no1_at_5857=f5857, no1_at_8265=f8265, verdict=v)]).to_csv(
    RESULTS / "E12A2_verdict.csv", index=False)
