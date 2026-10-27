"""E12-A: NO1 디지털 크롭(좌 오프셋 2408px, 1:1)에서 끝열 스파이크가 어디에 나오는지 본다.
센서·렌즈·장면에 붙어 있으면 NO1의 x ≈ 8265-2408 = 5857, 출력 좌표에 붙어 있으면 x ≈ 8265.
판정 기준은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 2절(결과 보기 전 고정)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from paths import RESULTS, FIGURES, UM_PER_PX
from e12_edges import TOL, W, CROP_DX, POS, ROWS, main, window_edges, load


def runs(dev, thr=-0.5, minlen=2):
    """dev <= thr가 minlen열 이상 이어지는 구간 [(시작, 끝)] (인덱스, 끝 포함)."""
    low = np.nan_to_num(dev, nan=0.0) <= thr
    out, s = [], None
    for i, v in enumerate(list(low) + [False]):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s >= minlen: out.append((s, i - 1))
            s = None
    return out


def analyse(D, xs, xc):
    """같은 프로파일에 판정 기준 3가지를 적용한다.
    asrun : 실행 당시 코드(창 전체 최저점 + 주변 7열에서 -0.25 이하 두 열). 지시서와 다르다(Codex 리뷰 지적), 기록용.
    strict: 지시서 0절 문구 그대로. 창 전체 최저점 열이 dev<=-0.5가 2열 이상 이어지는 구간 안에 있을 것.
    target: 사후 변경. 창 전체 최저점이 아니라 목표 xc±TOL 안에 strict 구간이 있는지. 목표 밖 구간 수도 함께 낸다."""
    d = np.nanmedian(D, 0)
    dev = d - pd.Series(d).rolling(13, center=True, min_periods=5).median().values
    j = int(np.nanargmin(dev))
    R = runs(dev)
    narrow_old = np.sum(dev[max(0, j - 3):j + 4] <= -0.25) >= 2
    near = [r for r in R if any(abs(xs[k] - xc) <= TOL for k in range(r[0], r[1] + 1))]
    best = min(near, key=lambda r: dev[r[0]:r[1] + 1].min()) if near else None
    off = [r for r in R if r not in near]
    return dict(
        x_peak=int(xs[j]), dev_px=round(float(dev[j]), 2), dev_um=round(float(dev[j]) * UM_PER_PX),
        found_asrun=bool(dev[j] <= -0.5 and narrow_old),
        found_strict=any(r[0] <= j <= r[1] for r in R),
        found_target=best is not None,
        target_x=None if best is None else int(xs[best[0] + int(np.argmin(dev[best[0]:best[1] + 1]))]),
        target_dev_px=None if best is None else round(float(dev[best[0]:best[1] + 1].min()), 2),
        n_off_target_runs=len(off), n_cols=len(xs)), d


summ, prof = [], []
jobs = [(s, "zoom1", 8265, 5) for s in POS] + [("NO1", "NO1", 5857, 10), ("NO1", "NO1", 8265, 10)]
for s, kind, xc, n in jobs:
    D, used = [], 0
    for r in main[main.specimen == s].head(n).itertuples():
        g = load(r)
        e = window_edges(g, xc, ROWS[kind])
        if e is not None: D.append(e); used += 1
    xs = np.arange(xc - W, xc + W)
    if not D:
        summ.append(dict(specimen=s, window_center=xc, n_img=0, found_asrun=False, found_strict=False, found_target=False)); continue
    res, d = analyse(D, xs, xc)
    summ.append(dict(specimen=s, window_center=xc, n_img=used, **res))
    prof += [dict(specimen=s, window_center=xc, x=int(x), dev_median_px=float(v)) for x, v in zip(xs, d)]
S = pd.DataFrame(summ)
S["near_8265"] = (S.x_peak - 8265).abs() <= TOL
S["near_5857"] = (S.x_peak - (8265 - CROP_DX)).abs() <= TOL
S.to_csv(RESULTS / "E12A_summary.csv", index=False)
pd.DataFrame(prof).to_csv(RESULTS / "E12A_no1_frame.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(S.to_string(index=False))


def verdict(variant):
    """variant: asrun|strict|target. asrun·strict는 창 전체 최저점 위치(x_peak)가 목표 ±TOL 안일 때만 센다."""
    pos = S[S.specimen.isin(POS)]
    if variant == "target": hit = pos.found_target
    else: hit = pos[f"found_{variant}"] & pos.near_8265
    repro = int(hit.sum())
    def at(xc, near_col):
        q = S[(S.specimen == "NO1") & (S.window_center == xc)].iloc[0]
        return bool(q.found_target) if variant == "target" else bool(q[f"found_{variant}"] and q[near_col])
    f5857, f8265 = at(5857, "near_5857"), at(8265, "near_8265")
    if repro < 4: v = "양성 대조 미재현 — 방법 문제, 중단"
    elif f5857 and not f8265: v = "센서·렌즈·장면 쪽 (출력 좌표 고정 아님)"
    elif f8265 and not f5857: v = "출력 좌표에 붙은 원인 (JPEG/ISP 후처리 쪽)"
    else: v = "판단 불가"
    return dict(variant=variant, positive_control=f"{repro}/5", no1_at_5857=f5857, no1_at_8265=f8265, verdict=v)


V = pd.DataFrame([verdict(v) for v in ("asrun", "strict", "target")])
V["status"] = ["실행 당시(지시서와 다름, 기록용)", "사전 기준(지시서 문구 그대로)", "사후 변경(결과를 본 뒤)"]
V.to_csv(RESULTS / "E12A_verdict.csv", index=False)
print(V.to_string(index=False))
import platform, matplotlib as _m
(RESULTS / "E12_env.txt").write_text(
    f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\nPillow {Image.__version__}\n"
    f"matplotlib {_m.__version__}\nseed: 난수 사용 없음\n", encoding="utf-8")

P = pd.DataFrame(prof)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s in POS:
    q = P[(P.specimen == s) & (P.window_center == 8265)]; ax[0].plot(q.x, q.dev_median_px, lw=.8, label=s)
q = P[(P.specimen == "NO1") & (P.window_center == 8265)]; ax[0].plot(q.x, q.dev_median_px, "k", lw=1.2, label="NO1")
ax[0].axvline(8265, c="r", ls=":"); ax[0].set_title("window at x=8265 (output coords)"); ax[0].legend(fontsize=7)
q = P[(P.specimen == "NO1") & (P.window_center == 5857)]; ax[1].plot(q.x, q.dev_median_px, "k", lw=1.2)
ax[1].axvline(5857, c="r", ls=":"); ax[1].set_title("NO1 window at x=5857 (= 8265-2408)")
for a in ax: a.set_xlabel("x (px)"); a.set_ylabel("top edge dev (px)")
fig.tight_layout(); fig.savefig(FIGURES / "E12A_no1_frame.png", dpi=110)
