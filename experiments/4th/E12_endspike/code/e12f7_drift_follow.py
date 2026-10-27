"""E12-F7: 스파이크 위치가 밝은 띠 질감의 이동을 따라가는가(장면 쪽) 사진 좌표에 고정인가(카메라 쪽).
10/21 본시험 NO2~NO12(110장) + 재장착 NO7~NO12(30장)만. 규칙은 docs/EXPERIMENTS_E12_endspike_for_ClaudeCode.md 9절(실행 전 고정)."""
import numpy as np
import pandas as pd
from paths import RESULTS, FIGURES, PROFILES as CACHE, RES1021, UM_PER_PX
from e12_edges import load, first_cross

XS = np.arange(8225, 8306)
TEX = (3128, 3308, 8100, 8430)           # 절대 y0,y1, x0,x1: 시편이 없는 밝은 띠
SP_WIN, DEPTH_MIN_PX, PEAK_MIN = (8259, 8272), 0.5, 0.30      # PEAK_MIN: 정렬 후 질감 정규화 상관(참고: 본시험 NO2~NO12 정렬 전 상관 0.53~0.80)
SLOPE_FOLLOW, SLOPE_FIXED, MIN_RANGE_PX, MIN_GROUPS, B, SEED = (0.5, 1.5), (-0.5, 0.5), 1.0, 8, 5000, 42


def tex(g):
    y0, y1, x0, x1 = TEX; a = g[y0:y1, x0:x1].astype(float); a = a - a.mean()
    return a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))


def tex_shift(ref, img):
    """img의 질감이 ref보다 (오른쪽·아래로) 얼마나 이동했나(px, 서브픽셀). 양수 = 오른쪽/아래. 반환: (dx, dy, 최대 정규화 상관).
    위상상관의 포물선 보간은 작은 이동(0.5px)을 절반으로 과소추정해서(합성 이동으로 확인) 쓰지 않는다.
    대신 ref를 푸리에 이동시킨 것과 img의 정규화 상관을 격자(dx 0.05px, dy 0.1px)에서 최대화하고 dx를 포물선으로 다듬는다."""
    ny, nx = ref.shape; fy, fx = np.fft.fftfreq(ny)[:, None], np.fft.fftfreq(nx)[None, :]
    Rf, If = np.fft.fft2(ref), np.fft.fft2(img); den = np.sqrt((np.abs(Rf) ** 2).sum() * (np.abs(If) ** 2).sum())
    cross = Rf * np.conj(If)
    # 거친 정수 이동으로 탐색 중심을 잡는다(위상상관 peak, 부호는 뒤집음)
    Rp = cross / (np.abs(cross) + 1e-9); r = np.fft.ifft2(Rp).real; iy, ix = np.unravel_index(r.argmax(), r.shape)
    cx = -(ix - nx if ix > nx / 2 else ix); cy = -(iy - ny if iy > ny / 2 else iy)
    dxs, dys = cx + np.arange(-1.5, 1.501, 0.05), cy + np.arange(-0.5, 0.501, 0.1)
    ph = lambda dx, dy: np.exp(-2j * np.pi * (fx * dx + fy * dy))
    C = np.array([[float((cross * ph(dx, dy)).real.sum() / den) for dx in dxs] for dy in dys])
    j, i = np.unravel_index(C.argmax(), C.shape)
    if 0 < i < len(dxs) - 1:
        l, c, h = C[j, i - 1], C[j, i], C[j, i + 1]; dx = dxs[i] + 0.05 * (l - h) / (2 * (l - 2 * c + h) + 1e-12)
    else: dx = dxs[i]
    return float(dx), float(dys[j]), float(C[j, i])


def spike_of(med):
    """윗엣지 추적선(장 방향 중앙값, 상대 y)에서 스파이크 깊이(px, 8265~8266 평균 편차)와 위치(무게중심 x). 기준선 = 롤링 15열 중앙값."""
    dev = med - pd.Series(med).rolling(15, center=True, min_periods=5).median().values
    m = (XS >= SP_WIN[0]) & (XS <= SP_WIN[1]); w = np.clip(-dev[m], 0, None)
    depth = float(dev[(XS >= 8265) & (XS <= 8266)].mean())
    return depth, (float((XS[m] * w).sum() / w.sum()) if w.sum() > 0 else np.nan)


def slope_ci(dx, xc, seed=SEED):
    rng = np.random.default_rng(seed); n = len(dx); sl = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.ptp(dx[i]) > 0: sl.append(np.polyfit(dx[i], xc[i], 1)[0])
    return float(np.polyfit(dx, xc, 1)[0]), tuple(np.percentile(sl, [2.5, 97.5]))


def verdict(n, rng_px, slope, ci):
    if n < MIN_GROUPS: return f"판단 불가 (유효 그룹 {n} < {MIN_GROUPS})"
    if rng_px < MIN_RANGE_PX: return f"판단 불가 (질감 이동 범위 {rng_px:.2f}px < {MIN_RANGE_PX}px)"
    if SLOPE_FOLLOW[0] <= ci[0] and ci[1] <= SLOPE_FOLLOW[1]:
        return "스파이크가 질감의 이동을 따라감 — 사진 좌표에 고정된 원인(센서·JPEG·ISP·렌즈 고정 패턴)에 불리, 장면(질감) 쪽 후보, 원인 판단 보류"
    if SLOPE_FIXED[0] <= ci[0] and ci[1] <= SLOPE_FIXED[1]:
        return "스파이크가 사진 좌표에 고정 — 질감 이동을 따르지 않음, 장면 질감 가설에 불리, 카메라 쪽 후보, 원인 판단 보류"
    return "판단 불가 (기울기 구간이 두 가설 범위에 들지 않음)"


def self_test():
    rng = np.random.default_rng(1); from numpy.fft import fft2, ifft2, fftfreq
    a = np.real(ifft2(fft2(rng.normal(size=(180, 330))) * np.exp(-((fftfreq(180)[:, None]) ** 2 + (fftfreq(330)[None, :]) ** 2) / (2 * .05 ** 2))))
    w = np.outer(np.hanning(180), np.hanning(330))
    for true in (1.7, -1.7, 0.5, 0.2, 1.0, 2.6):
        b = np.real(ifft2(fft2(a) * np.exp(-2j * np.pi * fftfreq(330)[None, :] * true)))
        dx, _, pk = tex_shift(a * w, b * w); assert abs(dx - true) < 0.08 and pk > 0.95, (true, dx, pk)


if __name__ == "__main__":
    self_test()
    z = np.load(CACHE, allow_pickle=True); cen = {f: int(c) for f, c in zip(z["filename"].astype(str), z["top_center"])}
    meta = pd.read_csv(RES1021 / "meta_1021_all.csv", encoding="utf-8-sig")
    sel = meta[(meta["단계"].str.startswith(("본시험", "재장착"))) & (meta.specimen != "NO1")]
    assert len(sel) == 140 and all(f in cen for f in sel.filename), "입력 불완전(본시험 110 + 재장착 30 = 140장)"
    ref = None; recs = []
    for (stage, spec), q in sel.groupby(["단계", "specimen"], sort=False):
        D, DX, PK, T = [], [], [], []
        for r in q.itertuples():
            g = load(r); c = cen[r.filename]
            e = np.array([(lambda v: np.nan if v is None else v + c - 60)(first_cross(g[c - 60:c + 61, x])) for x in XS]); D.append(e - np.nanmedian(e)); T.append(c)
            t = tex(g)
            if ref is None: ref = t
            dx, dy, pk = tex_shift(ref, t); DX.append(dx); PK.append(pk)
        depth, xc = spike_of(np.nanmedian(D, 0))
        half = [spike_of(np.nanmedian(D[k::2], 0))[1] for k in (0, 1)]
        recs.append(dict(stage=stage[:3], specimen=spec, n_img=len(q), time=q.시각.iloc[0], edge_center_y=int(np.median(T)), texture_dx_px=round(float(np.median(DX)), 2),
                         texture_peak=round(float(np.median(PK)), 2), spike_depth_um=round(depth * UM_PER_PX), spike_x=round(xc, 2),
                         spike_x_half_diff=round(abs(half[0] - half[1]), 2)))
    G = pd.DataFrame(recs)
    G["use"] = (G.spike_depth_um / UM_PER_PX <= -DEPTH_MIN_PX) & (G.texture_peak >= PEAK_MIN) & G.spike_x.notna()
    U = G[G.use]; dx, xc = U.texture_dx_px.values, U.spike_x.values
    slope, ci = slope_ci(dx, xc); rng_px = float(np.ptp(dx)); v = verdict(len(U), rng_px, slope, ci)
    sub = lambda m: (lambda d: np.polyfit(d.texture_dx_px, d.spike_x, 1)[0] if len(d) >= 3 and np.ptp(d.texture_dx_px) > 0 else np.nan)(U[m])
    tmin = (pd.to_datetime(U.time, format="%H:%M:%S") - pd.to_datetime(U.time, format="%H:%M:%S").min()).dt.total_seconds() / 60
    G.to_csv(RESULTS / "E12F7_groups.csv", index=False)
    pd.DataFrame([dict(n_groups=len(U), dx_range_px=round(rng_px, 2), slope=round(slope, 3), ci_lo=round(ci[0], 3), ci_hi=round(ci[1], 3),
                       slope_main_only=round(sub(U.stage == "본시험"), 3), slope_remount_only=round(sub(U.stage == "재장착"), 3),
                       slope_vs_time_px_per_min=round(float(np.polyfit(tmin, xc, 1)[0]), 4), corr_dx_time=round(float(np.corrcoef(dx, tmin)[0, 1]), 3),
                       median_split_half_diff_px=round(float(U.spike_x_half_diff.median()), 2), verdict=v)]).to_csv(RESULTS / "E12F7_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30); print(G.to_string(index=False)); print(f"\n사용 {len(U)}개 그룹, 질감 이동 범위 {rng_px:.2f}px, 기울기 {slope:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]\n판정: {v}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.5, 5)); 
    for st, mk in (("본시험", "o"), ("재장착", "s")):
        q = U[U.stage == st]; ax.scatter(q.texture_dx_px, q.spike_x, marker=mk, label=st, s=55)
        for _, r in q.iterrows(): ax.annotate(r.specimen, (r.texture_dx_px, r.spike_x), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xx = np.array([dx.min(), dx.max()]); ax.plot(xx, np.polyval(np.polyfit(dx, xc, 1), xx), "k-", lw=1, label=f"기울기 {slope:+.2f}")
    ax.plot(xx, xc.mean() + (xx - dx.mean()), "g:", label="기울기 1 (질감을 따라감)"); ax.plot(xx, [xc.mean()] * 2, "r:", label="기울기 0 (사진 좌표 고정)")
    ax.set_xlabel("질감 이동 (px, 오른쪽 +, NO2 기준)"); ax.set_ylabel("스파이크 위치 x (px)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "E12F7_drift_follow.png", dpi=110)
