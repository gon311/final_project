"""사후 서술(사전 등록 아님): 재장착 그룹이 F7에서 전부 제외된 이유와, 재장착 그룹의 스파이크 위치를 세 가설의 예측과 견준다.
F7은 스파이크 깊이를 x 8265~8266에서만 재서, 질감을 따라 옮겨 간 스파이크는 정의상 제외된다는 설계 약점이 있다."""
import numpy as np
import pandas as pd
from paths import RESULTS, PROFILES as CACHE, UM_PER_PX
from e12_edges import load, first_cross
from e12f7_drift_follow import XS

G = pd.read_csv(RESULTS / "E12F7_groups.csv"); M = G[G.stage == "본시험"]; U = M[M.use]
tmin = lambda s: (pd.to_datetime(pd.Series(list(s)), format="%H:%M:%S") - pd.to_datetime("15:35:54", format="%H:%M:%S")).dt.total_seconds().values / 60
sl_t, ic_t = np.polyfit(tmin(U.time), U.spike_x, 1); sl_d, ic_d = np.polyfit(U.texture_dx_px, U.spike_x, 1)
meta = pd.read_csv(__import__("paths").RES1021 / "meta_1021_all.csv", encoding="utf-8-sig")
z = np.load(CACHE, allow_pickle=True); cen = {f: int(c) for f, c in zip(z["filename"].astype(str), z["top_center"])}
rows = []
for r0 in G[G.stage == "재장착"].itertuples():
    q = meta[(meta["단계"].str.startswith("재장착")) & (meta.specimen == r0.specimen)]
    D = []
    for r in q.itertuples():
        g = load(r); c = cen[r.filename]
        e = np.array([(lambda v: np.nan if v is None else v + c - 60)(first_cross(g[c - 60:c + 61, x])) for x in XS]); D.append(e - np.nanmedian(e))
    med = np.nanmedian(D, 0); dev = med - pd.Series(med).rolling(15, center=True, min_periods=5).median().values
    m = (XS >= 8255) & (XS <= 8285); j = int(np.nanargmin(np.where(m, dev, np.inf)))
    rows.append(dict(specimen=r0.specimen, time=r0.time, texture_dx_px=r0.texture_dx_px, texture_peak=r0.texture_peak,
                     deepest_x=int(XS[j]), deepest_um=round(float(dev[j] * UM_PER_PX)), depth_8265_6_um=round(float(dev[(XS >= 8265) & (XS <= 8266)].mean() * UM_PER_PX)),
                     pred_follow_texture=round(ic_d + sl_d * r0.texture_dx_px, 1), pred_time_trend=round(ic_t + sl_t * tmin([r0.time])[0], 1), pred_image_fixed=8265.5))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "E12_peek_remount.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30); print(R.to_string(index=False))
print("\n참고: 본시험 스파이크 깊이(x 8265~8266, um):", dict(zip(M.specimen, M.spike_depth_um)))
