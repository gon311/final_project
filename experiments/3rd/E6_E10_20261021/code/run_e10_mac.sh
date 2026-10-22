#!/bin/bash
# 10/21 3차 촬영 E10 전체 실행 (Mac, 팀 저장소의 .venv 사용, MPS 자동 사용)
# 사용: bash experiments/3rd/E6_E10_20261021/code/run_e10_mac.sh
# 원본 E10/E7/E6 코드는 수정하지 않는다. 결과: results/E10/, 체크포인트: cache/E10/ (.gitignore 대상)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(cd ../../2nd && pwd)"
PY="$REPO/.venv/bin/python"
export REPO2ND="$REPO" E10_OUT=E10 PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p results/E10
LOG="results/E10/run_log.txt"
{
  echo "start: $(date)"
  "$PY" -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
  caffeinate -i "$PY" code/e10_1021.py pretrain
  caffeinate -i "$PY" code/e10_1021.py finetune
  "$PY" code/e10_1021.py evaluate
  echo "end: $(date)"
} 2>&1 | tee "$LOG"
