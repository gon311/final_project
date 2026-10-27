"""E11-F 재장착 변동 vs 거칠기 (PREREG 7절, 서술만·검정 없음). 실행: python3 e11f_remount.py"""
import numpy as np
import pandas as pd
from paths import RES1021, RESULTS

SPECS = [f"NO{k}" for k in range(7, 13)]
PIN_D = 16.0
df = pd.read_csv(RES1021 / "values_by_image_1021.csv", encoding="utf-8-sig")
main = df[df.specimen != "NO1"]
scale = PIN_D / main[main.mount == 0].pin_cross50_px.median()  # mm/px, 10/21 PREREG와 같은 정의
mv = main[main.specimen.isin(SPECS)].groupby(["specimen", "mount"]).agg(
    b=("bearing_cross50_px", "median"), p=("pin_cross50_px", "median"), n_img=("filename", "count")).reset_index()
assert mv.groupby("specimen").size().eq(6).all(), "시편당 장착 6점이 아님"
mv["ratio_mm"] = PIN_D * mv.b / mv.p
g = mv.groupby("specimen")
rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")
out = pd.DataFrame({
    "n_mounts": g.size(), "sd_raw_um": g.b.std() * scale * 1000, "sd_ratio_um": g.ratio_mm.std() * 1000,
    "sd_pin_px": g.p.std(), "Rz": rough.Rz.loc[SPECS], "Ra": rough.Ra.loc[SPECS], "Rz_Ra": rough.Rz_Ra.loc[SPECS],
}).loc[SPECS].sort_values("Rz")
med = out[["sd_raw_um", "sd_ratio_um", "sd_pin_px"]].median()
# 10/21 PREREG 부록 값 재현: 재장착 σ 중앙값 원시 5.2µm / 핀 비율 6.4µm
assert abs(med.sd_raw_um - 5.2) < 0.05 and abs(med.sd_ratio_um - 6.4) < 0.05, f"재현 실패 {med.to_dict()}"
res = pd.concat([out.reset_index(names="specimen"),
                 pd.DataFrame([dict(specimen="중앙값(6시편)", **med.to_dict())])], ignore_index=True)
res.to_csv(RESULTS / "E11F_remount.csv", index=False)
pd.set_option("display.width", 200)
print(f"scale={scale:.7f} mm/px"); print(res.round(3).to_string(index=False))
