"""roughness_raw.csv -> results/roughness_clean.csv (대응: NOk->NOk, NG1->NO11, NG2->NO12)."""
import pandas as pd
from paths import E11, RESULTS

raw = pd.read_csv(E11 / "data" / "roughness_raw.csv", header=2, encoding="utf-8-sig")
raw = raw[raw["구분"].astype(str).str.match(r"^(NO\d+|NG\d)$")]
mapping = {"NG1": "NO11", "NG2": "NO12"}
df = pd.DataFrame({
    "rough_id": raw["구분"].values,
    "specimen_1021": [mapping.get(s, s) for s in raw["구분"]],
    "Ra": raw["Ra"].astype(float).values,
    "Rq": raw["Rq"].astype(float).values,
    "Rz": raw["Rz"].astype(float).values,
})
df["Rz_Ra"] = df["Rz"] / df["Ra"]
df["in_analysis"] = df["specimen_1021"] != "NO1"  # NO1은 촬영 설정이 달라 10/21 분석과 같이 제외
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "roughness_clean.csv", index=False)
print(df.to_string(index=False))
