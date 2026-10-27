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
# 열별 프로파일 캐시(저장소 밖, 읽기 전용): 사진별 윗엣지 중심 y(top_center)를 얻는 데만 쓴다.
PROFILES = Path(os.environ.get(
    "E12_PROFILES",
    "/Volumes/Lexar/Final/Claude outputs/20261021_3차_E6_E10/_tmp/profiles_1021.npz"))
