"""캐시 + 메타 -> 열 단위 긴 표 cache/edges_long.csv.gz (y = center + label)."""
import numpy as np
import pandas as pd
from paths import PROFILES, RES1021, CACHE

z = np.load(PROFILES, allow_pickle=True)
df = pd.DataFrame({
    "filename": z["filename"].astype(str),
    "specimen": z["specimen"].astype(str),
    "x": z["x"].astype(int),
    "y_top": z["top_center"] + z["top_label"],
    "y_bot": z["bot_center"] + z["bot_label"],
})
meta = pd.read_csv(RES1021 / "meta_1021_all.csv", encoding="utf-8-sig")[["filename", "단계", "mount", "시각"]]
out = df.merge(meta, on="filename", how="left", validate="m:1")
assert out["단계"].notna().all(), "메타에 없는 filename"
out = out[["filename", "specimen", "단계", "mount", "시각", "x", "y_top", "y_bot"]]
CACHE.mkdir(exist_ok=True)
out.to_csv(CACHE / "edges_long.csv.gz", index=False)
print(out.shape, out["filename"].nunique(), "images")
print(out.groupby("단계")["filename"].nunique())
