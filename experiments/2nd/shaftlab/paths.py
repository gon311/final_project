"""2nd 실험 공용 경로 상수. 저장소 밖 데이터·코드는 모두 읽기 전용으로만 참조한다 (0.5절 규칙 1)."""
import os

DATA_ROOT = os.environ.get("SHAFT_DATA_ROOT", "/Volumes/Lexar/Final/_private/2026_10_08")

IMAGE_DIR = os.path.join(DATA_ROOT, "20261008_촬영데이터")
CLASSIFICATION_CSV = os.path.join(IMAGE_DIR, "분류표_full_result.csv")

EDA_DIR = os.path.join(DATA_ROOT, "EDA_20261008")
EDA_SCRIPTS_DIR = os.path.join(EDA_DIR, "scripts")
EDA_MEASURE_SCRIPT = os.path.join(EDA_SCRIPTS_DIR, "eda_measure.py")
EDA_MEASURE_RAW_CSV = os.path.join(EDA_DIR, "data", "eda_measure_raw.csv")
PIN_GAUGE_RAW_CSV = os.path.join(EDA_DIR, "data", "pin_gauge_raw.csv")

# 팀원A 측정 엔진 (읽기 전용 참조, 수정하지 않음)
HEEYEON_ENGINE_V3 = (
    "/Volumes/Lexar/Final/Heeyeon/Shaft_OD_Measurement/12_System_Integration/"
    "measurement/measurement_engine_EDGE_V3_REFLECTION.py"
)
HEEYEON_ENGINE_V2 = (
    "/Volumes/Lexar/Final/Heeyeon/Shaft_OD_Measurement/12_System_Integration/"
    "measurement/measurement_engine.py"
)

# 멘토 MKD 코드 (읽기 전용 참조)
MKD_MENTOR_DIR = "/Volumes/Lexar/Final/_private/fwd31/멘토수정코드"
MKD_ORIGINAL_DIR = (
    "/Volumes/Lexar/Final/_private/fwd31/"
    "Knowledge_Distillation_AD-main_논문제공코드(바로실행안됨)"
)

# 팀원B 분석 (읽기 전용·대조용, import·실행 금지. 0.4절)
MINWOOK_ROOT = "/Volumes/Lexar/Final/Minwook/shaft-od-vision/experiments/20261008_2차촬영"

# 현장 메모 사진 (분류표에 없음, 항상 측정 대상에서 제외. 0.3절)
EXCLUDED_FILES = {"20261008_155336.jpg"}
