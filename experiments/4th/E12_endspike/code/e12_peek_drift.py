"""사후 예비 관찰(사전 등록 아님): 밝은 띠 질감이 촬영 순서에 따라 이동하는 동안 8265 스파이크의 위치도 따라 움직이는가.
질감 이동(양수 = 오른쪽) = 절대 y 3128~3308, x 8100~8430 영역을 NO2 첫 사진과 위상상관(서브픽셀). 스파이크 위치 = 윗엣지 추적선의 기준선 대비 음의 편차 무게중심(x 8259~8272)."""
import numpy as np
import pandas as pd
from paths import RESULTS, PROFILES as CACHE, UM_PER_PX
from e12_edges import main, load, first_cross

SP = [f"NO{i}" for i in range(2, 13)]; XS = np.arange(8225, 8306)
z = np.load(CACHE, allow_pickle=True); cen = {f: int(c) for f, c in zip(z["filename"].astype(str), z["top_center"])}
rows = {s: [r for r in main.itertuples() if r.specimen == s] for s in SP}


def tex(g):
    a = g[3128:3308, 8100:8430].astype(float); a = a - a.mean(); return a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))


def shift(a, b):
    R = np.fft.fft2(a) * np.conj(np.fft.fft2(b)); R /= np.abs(R) + 1e-9; r = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(r.argmax(), r.shape)
    def sub(i, n, axis):
        l, c, h = (r[(i - 1) % n, ix], r[iy, ix], r[(i + 1) % n, ix]) if axis == 0 else (r[iy, (i - 1) % n], r[iy, ix], r[iy, (i + 1) % n])
        v = i + (l - h) / (2 * (l - 2 * c + h) + 1e-12); return v - n if v > n / 2 else v
    return sub(ix, r.shape[1], 1), sub(iy, r.shape[0], 0)


rec, base = [], None
for s in SP:
    T = []
    for k, r in enumerate(rows[s]):
        g = load(r); c = cen[r.filename]
        e = np.array([(lambda v: np.nan if v is None else v + c - 60)(first_cross(g[c - 60:c + 61, x])) for x in XS]); T.append(e - np.nanmedian(e))
        if k == 0:
            tx = tex(g); base = tx if base is None else base; dx, dy = shift(base, tx); dx, dy = -dx, -dy   # shift()의 부호는 실제 이동과 반대임을 합성 이동으로 확인 → 뒤집어 '양수 = 오른쪽/아래로 이동(NO2 기준)'
    med = np.nanmedian(T, 0); dev = med - pd.Series(med).rolling(15, center=True, min_periods=5).median().values
    m = (XS >= 8259) & (XS <= 8272); w = np.clip(-dev[m], 0, None)
    rec.append(dict(specimen=s, time=rows[s][0].시각, texture_dx_px=round(dx, 2), texture_dy_px=round(dy, 2),
                    spike_centroid_x=round(float((XS[m] * w).sum() / w.sum()), 2) if w.sum() > 0 else np.nan,
                    spike_depth_um=round(float(-dev[(XS >= 8265) & (XS <= 8266)].mean() * UM_PER_PX))))
D = pd.DataFrame(rec); D.to_csv(RESULTS / "E12_peek_drift.csv", index=False)
d = D[D.specimen != "NO8"].dropna()                                      # NO8은 스파이크가 뚜렷하지 않아 제외
sl, ic = np.polyfit(d.texture_dx_px, d.spike_centroid_x, 1)
loo = [np.polyfit(d.texture_dx_px.drop(i), d.spike_centroid_x.drop(i), 1)[0] for i in d.index]
print(D.to_string(index=False)); print(f"\n스파이크 위치 vs 질감 이동 기울기 = {sl:+.2f} (1이면 스파이크가 질감을 따라감, 0이면 사진 좌표에 고정), 하나씩 빼기 범위 {min(loo):+.2f}~{max(loo):+.2f}, n={len(d)}")
