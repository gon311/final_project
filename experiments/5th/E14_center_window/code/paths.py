"""E14 경로 한 곳 관리. 저장소 안 경로는 이 파일 기준 상대경로."""
from pathlib import Path

E14 = Path(__file__).resolve().parent.parent
REPO = E14.parents[2]
R1021 = REPO / "experiments" / "3rd" / "E6_E10_20261021"
RES1021 = R1021 / "results"
EDGES = REPO / "experiments" / "3rd" / "roughness" / "cache" / "edges_long.csv.gz"  # 없으면 roughness/code/load_edges.py
E6_CODE = REPO / "experiments" / "2nd" / "E6_calibration"
RESULTS = E14 / "results"
FIGURES = E14 / "figures"
SEED = 42
