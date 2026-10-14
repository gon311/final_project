# E1. 측정 모듈 분리 + 재현 확인

## 목적
이후 모든 2차 실험이 같은 측정 코드를 쓰도록, `DATA_ROOT/EDA_20261008/scripts/eda_measure.py`
(저장소 밖, 읽기 전용, 수정 안 함)의 `coarse_section`/`edge_subpx`/`measure_section`을
저장소 안 import 가능한 모듈 `experiments/2nd/shaftlab/measure.py`로 옮기고, 기존
`eda_measure_raw.csv` 수치가 그대로 재현되는지 확인한다.

## 한 일
1. `.gitignore`: `*.jpeg` 추가, R 템플릿의 `docs/` 무시 규칙을 주석 처리(지시서 `docs/EXPERIMENTS_20261008.md`가 git에 잡히도록).
2. `experiments/2nd/shaftlab/paths.py`: `SHAFT_DATA_ROOT` 환경변수(없으면 기본 `/Volumes/Lexar/Final/_private/2026_10_08`) + 팀원A 엔진·MKD·`MINWOOK_ROOT` 경로 상수.
3. `experiments/2nd/shaftlab/measure.py`: `coarse_section`/`edge_subpx`/`measure_section`을 산술 그대로 옮기고, 공개 인터페이스 `measure_bearing(gray_full, edge_method)`, `measure_pin(gray_full, edge_method)`, `extract_profiles(gray_full, section, half_window, step)`를 추가. `edge_method ∈ {"cross50", "grad_peak"}`은 `diameter_px`로 어느 값(d_line/d_peak_median)을 쓸지만 결정하고, 두 값 모두 반환 dict에 남겨 둬 기존 CSV 열과 1:1 비교할 수 있게 했다.
4. `experiments/2nd/requirements.txt` + `experiments/2nd/.venv`(커밋 안 함, `.gitignore`의 `.venv` 규칙에 이미 걸림).
5. `reproduce.py`: 131장 전부를 `measure_bearing`·`measure_pin`으로 다시 재서 기존 `eda_measure_raw.csv`(`B_d_line`, `B_d_peak_median`)와 `pin_gauge_raw.csv`(`P_d_line`, `P_d_peak`)를 이미지별로 비교한다. 파일 집합·실패(에러) 상태가 기존 CSV와 하나라도 다르거나 `|Δ| >= 1e-6px`인 항목이 있으면 **비정상 종료(exit 1)** 한다(Codex 1차 리뷰 반영, 아래 표).
6. (선택, 4단계) `compare_teamB.py`: 우리 cross50(`d_line`) 결과를 팀원B `1008_field/_out/full_rows.csv`의 `샤프트px`(50% 경계, 다른 측정 열 범위)와 파일명으로 맞춰 대조.

## 결과 요약
- **재현 (3단계, 베어링):** 131장 중 130장에서 양쪽 다 측정 성공. `max|Δ d_line| = 7.3e-12 px`, `max|Δ d_peak_median| = 0.0 px` → 완료 기준(`|Δ| < 1e-6px`) **PASS**. 남은 1장(`20261008_155336.jpg`, 현장 메모 사진, 0.3절에 따라 항상 제외 대상)은 원본과 우리 모듈 둘 다 `bearing seg too short`로 동일하게 실패한다.
- **재현 (핀, Codex 1차 리뷰 반영):** `pin_gauge_raw.csv` 기준 130장 전부 양쪽 다 측정 성공. `max|Δ P_d_line| = 3.2e-12 px`, `max|Δ P_d_peak| = 0.0 px` → **PASS**.
- **파일 집합·오류 상태 강제 검사(Codex 1차 리뷰 반영):** 이미지 폴더 파일 집합, 베어링 측정 실패 파일 집합(`paths.EXCLUDED_FILES`와 일치해야 함), 핀 대상 파일 집합(분류표 `시편` 열이 있는 파일과 일치해야 함)을 모두 하드 assert로 확인한다. 하나라도 어긋나면 `reproduce.py`가 FAIL 메시지를 찍고 `sys.exit(1)`한다 — 실제로 `measure_bearing`에 +0.01px 오차를 주입해 FAIL(exit 1, diff 리포트)이 나오는지 확인한 뒤 원복했다.
- **`extract_profiles` 열 수 커버리지(Codex 1차 리뷰 반영):** 샘플 3장에서 `n_profiles == n_cols`(74==74) 확인. 수정 전에는 기본 `step=2`가 이미 2px 간격인 `x_per_col`을 다시 인덱스 기준으로 2칸씩 건너뛰어 **열의 절반(37/74)만 추출하는 버그**가 있었다(아래 표 참고).
- **팀원B 대조 (4단계, 참고용):** 130장 매칭. `ours cross50 d_line − teamB 샤프트px` 중앙값 +0.75px (범위 +0.17\~+3.68px), 묶음별 중앙값 +0.29\~+3.26px로 셋업마다 다르다. 측정 열 범위(`x0,x1`)가 서로 달라(0.4절 "측정 열 범위가 달라 차이가 있어도 실패가 아니다") 나는 체계 차이로 보이고, 묶음 3·4·5에서 차이가 커 측정 열 범위가 카메라 이동 구간과 겹쳤을 가능성이 있다(E5에서 다룸).

## Codex 1차 adversarial-review 반영 표
| 지적 | 반영 여부 | 근거 |
|---|---|---|
| `extract_profiles`가 기본 설정에서 열의 절반만 추출 | **반영함** | `measure_bearing`으로 측정한 섹션에 기본 파라미터로 `extract_profiles`를 돌려보니 `n_profiles=37`, `n_cols=74`로 정확히 절반이었다(코드로 재현). `x_per_col`은 이미 2px 간격인데 `extract_profiles`가 그 인덱스를 다시 `step`만큼 건너뛴 게 원인 — 지시서 0.4절의 E7 데이터량 추정("130×75×2≈1.95만 개")도 75열/장을 가정하므로 **문서가 아니라 코드가 틀렸다는 근거가 명확**했다. `measure.py`를 고쳐 `step`을 "x_per_col 인덱스 보폭"이 아니라 "실제 이미지 x 픽셀 보폭"으로 재정의했고, 수정 후 같은 이미지에서 `n_profiles=74=n_cols`로 일치함을 확인했다. `reproduce.py`에 `check_extract_profiles_coverage()`를 추가해 재발 시 자동으로 잡히게 했다(`results/extract_profiles_check.csv`). |
| `measure_pin`이 재현 검증에서 빠짐 | **반영함** | `reproduce.py`에 `pin_gauge_raw.csv`의 `P_d_line`/`P_d_peak` 대조를 추가했다(130장, `max|Δ|` 둘 다 1e-11px대). 핀 대상 파일 집합도 "분류표 `시편` 열이 있는 파일"과 일치하는지 하드 assert로 강제한다. |
| 재현 PASS 조건이 실패·누락 행을 무시함 | **반영함** | 이미지 폴더 파일 집합과 `eda_measure_raw.csv` 파일 집합이 정확히 같은지, 베어링 실패 파일 집합이 `paths.EXCLUDED_FILES`와 같은지, 핀 대상 파일 집합이 분류표 기준과 같은지, 이미지별 성공/실패 상태가 원본과 같은지를 모두 하드 assert로 바꿨다. 하나라도 어긋나면 `sys.exit(1)`. `measure_bearing`에 +0.01px 오차를 주입해 실제로 FAIL(exit 1)이 나는지 확인한 뒤 원복했다(아래 "검증" 참고). |

## Codex 2차 review 반영 표
| 지적 | 반영 여부 | 근거 |
|---|---|---|
| [P1] `opencv-python-headless` 버전 핀이 PEP 440상 설치 불가능(`==5.0.0`인데 실제 패키지는 `5.0.0.93`) | **반영함** | `pip freeze`로 venv 안 실제 버전이 `opencv-python-headless==5.0.0.93`임을 확인(`cv2.__version__`은 `5.0.0`만 보여줘서 혼동했었다). `requirements.txt`를 `5.0.0.93`으로 고쳤다. 나머지 세 핀(`numpy==2.4.6`, `pandas==3.0.6`, `pillow==12.3.0`)은 `pip._vendor.packaging`으로 PEP 440 유효성과 자기 자신과의 매칭을 확인해 문제없음을 확인했다. 수정한 `requirements.txt`로 완전히 새 venv(`/tmp`)를 만들어 `pip install -r requirements.txt` → `pip check`(no broken requirements) → import 버전 확인까지 재현했다. |
| [P2] `reproduce.py`가 예외 없이 NaN을 내는 경우를 성공으로 오판할 수 있음(`NaN >= tol`이 항상 False) | **반영함** | `all_finite()` 헬퍼를 추가해 베어링·핀 양쪽에서 **새 측정값과 기존 CSV 기준값 양쪽 모두**를 `np.isfinite`로 검사하고, 그 결과로 성공/실패를 판정하도록 바꿨다(예외 발생 여부가 아니라 값의 유한성이 기준). `measure_bearing`을 임시로 패치해 예외 없이 `diameter_px = NaN`을 돌려주게 만든 뒤 재실행해, 수정 전이라면 조용히 지나갔을 상황에서 130장 전부 "성공/실패 상태가 기존 CSV와 다름"으로 FAIL(exit 1)이 나는 걸 확인했고, 이후 패치를 원복해 다시 PASS(exit 0)를 확인했다. |

## Codex 3차 review 반영 표
| 지적 | 반영 여부 | 근거 |
|---|---|---|
| [P2] 이미지 폴더에 CSV에 없는 파일이 섞이면 `check_file_sets`가 불일치를 FAIL로 기록만 하고 계속 진행해, `measure_one()`이 `bearing_orig.loc[fn]`에서 `KeyError`로 죽음(의도한 제어된 실패 보고가 아니라 비정상 크래시) | **반영함** | `main()`에서 `check_file_sets()` 직후 `problems`가 비어있지 않으면 **교집합을 추려 계속하지 않고** 바로 `report_and_exit()`로 FAIL 보고서를 찍고 `sys.exit(1)`하게 바꿨다(측정 루프를 아예 타지 않는다). 검증: 외부 이미지 폴더(`DATA_ROOT` 아래, 읽기 전용이지만 사용자 요청으로 임시 테스트)에 `ZZZZ_TEST_EXTRA.jpg` 빈 파일을 넣고 재실행 → `KeyError` 없이 `[FAIL] 이미지 폴더 파일 집합이 eda_measure_raw.csv와 다름: ... extra={'ZZZZ_TEST_EXTRA.jpg'}` 메시지와 `exit code 1`로 정상 종료함을 확인. 테스트 파일은 즉시 삭제하고 원래 131장 상태로 되돌린 뒤 PASS를 재확인했다. |

## 팀원B 결과와 대조
같은 항목(샤프트 px, 50% 경계) 있음 — **불일치(체계 차이), 환산 방식은 동일(둘 다 50% 교차, mm 환산 없음)**. 측정 열 범위가 다르므로(우리 `x0≈7892~8040`류, 팀원B `x0,x1` 별도 열) 직접 비교 대상이 아니다. 0.4절에 따라 체계 차이(중앙값)만 기록했다.

## 그림
없음(E1은 모듈화·재현 확인이라 그림 산출물이 지시서에 없음).

## 검증 (FAIL 경로 확인)
- **Codex 1차 (diff 임계값):** `reproduce.py`의 비정상 종료 로직이 실제로 작동하는지 확인하기 위해, `measure_bearing`의 `diameter_px`에 `+0.01px`를 임시로 더하는 패치를 적용하고 다시 돌렸다. 결과: 130장 전부에서 `|diff_d_line|=1.000e-02px >= 1e-6` FAIL 메시지가 출력되고 `exit code 1`로 종료했다.
- **Codex 2차 (isfinite 가드):** `measure_bearing`이 예외 없이 `diameter_px = NaN`을 돌려주도록 임시로 패치하고 다시 돌렸다. 결과: 130장 전부에서 "베어링 측정 성공/실패 상태가 기존 CSV와 다름(orig_failed=False, new_failed=True)" FAIL 메시지가 출력되고 `exit code 1`로 종료했다 — 수정 전 코드였다면 `diff_d_line`이 NaN이라 `NaN >= 1e-6`이 False가 되어 조용히 PASS로 지나갔을 상황이다.
- **Codex 3차 (파일 집합 불일치 → 제어된 종료):** `DATA_ROOT/20261008_촬영데이터/`(저장소 밖, 보통 읽기 전용)에 빈 `ZZZZ_TEST_EXTRA.jpg`를 임시로 추가하고 재실행 → `KeyError` 없이 `[FAIL] 이미지 폴더 파일 집합이 eda_measure_raw.csv와 다름: ... extra={'ZZZZ_TEST_EXTRA.jpg'}` 메시지와 `exit code 1`로 종료함을 확인했다. 테스트 파일은 곧바로 삭제해 131장 원상태로 되돌렸다(사용자가 직접 요청한 임시 테스트).
- 세 패치 모두 테스트 후 원본으로 복원했고(코드 패치는 `diff`로 바이트까지 동일함 확인, 이미지 폴더는 파일 개수 131장으로 복귀 확인), 매번 복원된 코드로 다시 돌려 `exit code 0`/PASS를 재확인했다. 이번 스테이징·커밋에는 복원된 코드만 들어간다.
- **requirements.txt:** `/tmp`에 완전히 새 venv를 만들어 `pip install -r requirements.txt` → `pip check`("No broken requirements found") → `cv2/numpy/pandas/PIL` import까지 재현해, 수정한 핀 버전으로 깨끗하게 설치되는지 확인했다.

## 한계와 다음 촬영에서 확인할 점
- 재현 확인은 **cross50/grad_peak 두 정의만** 봤다(E4에서 ESF 피팅 등 추가).
- `extract_profiles`는 E7에서 쓸 예정으로 만들었을 뿐, 이번 실험에서는 커버리지(열 수)만 확인했고 half_window 경계 처리나 half_window=40 외 다른 값에서의 동작은 검증하지 않았다.
- 13개 셋업, 재장착 반복 없음 — 이 데이터로 최종 성능을 주장하지 않는다(규칙 8).
- 팀원B 대조는 측정 열 범위가 달라 참고용일 뿐, teacher 선정이나 보정식 근거로 쓰지 않는다.

## 실행 명령 (재현용)
```bash
cd experiments/2nd
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd E1_module
python3 reproduce.py          # results/reproduce_diff.csv, results/extract_profiles_check.csv (실패 시 exit 1)
python3 compare_teamB.py      # results/vs_teamB.csv (선택)
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93 (`cv2.__version__`은 `5.0.0`만 보고하지만 실제 설치 패키지 버전은 `.93`까지다 — Codex 2차 리뷰 반영, `requirements.txt` 핀도 이 값으로 고쳤다)
- numpy 2.4.6
- pandas 3.0.6
- pillow 12.3.0
- torch: 사용하지 않음(E1은 PyTorch 불필요)
- 랜덤 시드: 사용 안 함(이번 실험은 결정론적 측정이라 난수 없음)

## 변경 파일 목록
```
M .gitignore
A docs/EXPERIMENTS_20261008.md   (이전에 docs/ 규칙으로 무시되던 지시서, 이번에 git이 잡게 됨)
A experiments/2nd/requirements.txt
A experiments/2nd/shaftlab/__init__.py
A experiments/2nd/shaftlab/paths.py
A experiments/2nd/shaftlab/measure.py
A experiments/2nd/E1_module/README.md
A experiments/2nd/E1_module/reproduce.py
A experiments/2nd/E1_module/compare_teamB.py
A experiments/2nd/E1_module/results/reproduce_diff.csv
A experiments/2nd/E1_module/results/extract_profiles_check.csv
A experiments/2nd/E1_module/results/vs_teamB.csv
```
(`experiments/2nd/.venv/`, `experiments/2nd/cache/`는 `.gitignore`에 걸려 스테이징 대상 아님. `experiments/2nd/E1_module/figures/`는 빈 디렉터리라 git에 잡히지 않음.)
5MB 초과 파일 없음(두 결과 CSV 모두 수십 KB 수준).

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- `extract_profiles`를 "x_per_col 인덱스 보폭"에서 "실제 x 픽셀 보폭"으로 고친 수정이 `half_window`가 40이 아닌 다른 값이나 핀(`PIN_COARSE`, 열 수가 훨씬 적음) 섹션에서도 여전히 `n_cols`와 맞는지 봐 달라(이번엔 베어링 섹션 3장만 샘플 검사했다).
- `reproduce.py`의 하드 assert(파일 집합, 실패 집합, 핀 대상 집합)가 EDA 코드나 분류표가 바뀌는 다음 촬영에서도 "진짜 문제"와 "정상적인 데이터 변화"를 잘 구분해 낼지(너무 엄격해서 매번 깨지지는 않을지) 봐 달라.
- `all_finite()` 가드를 베어링·핀 양쪽 다 새 측정값과 기존 CSV 기준값에 적용했는데, 혹시 다른 곳(예: `check_extract_profiles_coverage`의 `n_cols`/`n_profiles` 자체가 NaN이 될 수 있는 경로)에도 같은 문제가 남아 있는지 봐 달라.
- `.gitignore`에서 `docs/` 규칙을 주석 처리한 것이 R 패키지 문서 산출물(`docs/_build/`, `/site` 등 다른 규칙으로 이미 커버됨)에 영향을 주지 않는지 확인해 달라.
