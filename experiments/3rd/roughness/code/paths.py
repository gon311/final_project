"""E11 경로 한 곳 관리. 저장소 안 경로는 이 파일 기준 상대경로, 캐시만 저장소 밖(환경변수 E11_PROFILES로 덮어씀)."""
import os
from pathlib import Path

E11 = Path(__file__).resolve().parent.parent
REPO = E11.parents[2]
R1021 = REPO / "experiments" / "3rd" / "E6_E10_20261021"
RES1021 = R1021 / "results"
PROFILES = Path(os.environ.get(
    "E11_PROFILES",
    "/Volumes/Lexar/Final/Claude outputs/20261021_3차_E6_E10/_tmp/profiles_1021.npz"))
RESULTS = E11 / "results"
FIGURES = E11 / "figures"
CACHE = E11 / "cache"
SEED = 42
UM_PER_PX = 26.092  # V1 = 0.026092 mm/px
