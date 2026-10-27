"""E12 경로 한 곳 관리. 저장소 안은 이 파일 기준 상대경로, 사진 폴더만 저장소 밖(환경변수 E12_PHOTOS로 덮어씀)."""
import os
from pathlib import Path

E12 = Path(__file__).resolve().parent.parent
REPO = E12.parents[2]
RES1021 = REPO / "experiments" / "3rd" / "E6_E10_20261021" / "results"
PHOTOS = Path(os.environ.get("E12_PHOTOS", "/Volumes/Lexar/Final/_private/20261021_촬영데이터"))
RESULTS = E12 / "results"
FIGURES = E12 / "figures"
SEED = 42
UM_PER_PX = 26.092
