"""E13 경로 한 곳 관리 (이 파일 기준 상대경로)."""
from pathlib import Path

E13 = Path(__file__).resolve().parent.parent
REPO = E13.parents[2]
E11 = REPO / "experiments" / "3rd" / "roughness"
E11_CODE = E11 / "code"
EDGES = E11 / "cache" / "edges_long.csv.gz"
ROUGH = E11 / "results" / "roughness_clean.csv"
META = REPO / "experiments" / "3rd" / "E6_E10_20261021" / "results" / "values_by_image_1021.csv"
E11A_BY_SPEC = E11 / "results" / "E11A_by_specimen.csv"
RESULTS = E13 / "results"
FIGURES = E13 / "figures"
SEED = 42
UM_PER_PX = 26.092  # 0.026092 mm/px
MM_PER_PX = 0.026092
