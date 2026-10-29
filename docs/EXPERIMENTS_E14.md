# E14 중앙 구간 대표값 — Claude Code 작업 지시서

> 작성 2026-10-30 (rev1: Codex 리뷰 반영, PREREG 부록 A와 함께) · 저장소 `/Volumes/Lexar/Final/Dagyeom/git_upload/final_project` · 브랜치 `feature/5th_expr` · 작업 폴더 `experiments/5th/E14_center_window/`
> 규칙 문서 `docs/PREREG_E14.md`를 **먼저 끝까지 읽는다.** 판정 기준·지표·통계는 PREREG가 정하고, 이 문서는 구현 방법과 산출물을 정한다. 두 문서가 다르면 PREREG를 따르고 그 차이를 보고한다.
> 작업 규칙·보고 형식은 `docs/EXPERIMENTS_20261008.md` 0.5절(규칙 1\~14)·0.6절(저장소·커밋)을 따른다. 브랜치와 작업 위치는 위 값으로 바꿔 읽는다. 특히 규칙 3(셋업 단위 분할), 4(NG 학습 미사용), 10(평균 찍기 기준선)을 지킨다.
> **한 번에 단계 하나만 진행한다.** 단계가 끝나면 7절 형식으로 보고하고 멈춘다. 다음 단계는 사용자가 지시할 때 시작한다.
> **새 측정·재촬영은 없다.** 원본 이미지도 디코딩하지 않는다. 10/21 결과 CSV와 E11 열 캐시만 쓴다.

---

## 0. 경로와 환경

저장소 루트를 `REPO`라고 한다.

| 항목 | 경로 |
|---|---|
| 작업 폴더 | `REPO/experiments/5th/E14_center_window/` (이하 `E14`) |
| 규칙·지시서 | `REPO/docs/PREREG_E14.md`, `REPO/docs/EXPERIMENTS_E14.md`(이 문서) |
| 열 단위 엣지(저장소 밖 캐시에서 생성) | `REPO/experiments/3rd/roughness/cache/edges_long.csv.gz` (gitignore됨). 없으면 `experiments/3rd/roughness/code/load_edges.py`로 다시 만든다(원본 `profiles_1021.npz`, 환경변수 `E11_PROFILES`) |
| 이미지별 측정값·실측·핀 | `REPO/experiments/3rd/E6_E10_20261021/results/values_by_image_1021.csv` |
| E6 결과(재현 대조용) | `REPO/experiments/3rd/E6_E10_20261021/results/E6/calibration_loso_1021.csv` |
| 스케일·반복성 | `REPO/experiments/3rd/E6_E10_20261021/results/repeatability_ratio.json` |
| 캘리브레이션 함수(수정 금지, import만) | `REPO/experiments/2nd/E6_calibration/calibration.py` (`fit_fixed_scale`, `fit_linear`, `fit_const`, `judge`) |
| 원 측정 코드(참고) | `REPO/experiments/2nd/shaftlab/measure.py`(`_measure_window`, `robust_fit`) |

- **`experiments/2nd/`, `experiments/3rd/`, 캐시 원본은 수정하지 않는다.** 출력은 모두 `E14` 아래 `code/`, `results/`, `figures/`, `README.md`(마지막 단계의 종합 보고서)에 둔다.
- 경로는 `code/paths.py` 한 곳에 둔다. 저장소 안 경로는 `paths.py` 위치를 기준으로 한 상대경로로 만들고, 결과 CSV에는 절대경로를 쓰지 않는다.
- 코드는 `code/e14.py` 하나에 두고 `--step 0|1|2`로 단계를 나눈다(PREREG 7절).
- 환경: 사용자 Mac, CPU, numpy·pandas·matplotlib. 무작위화 검정은 512가지 교환 패턴을 모두 열거하는 결정적 계산이다. 검정력 모의실험은 시드 42로 고정한다. 패키지 버전을 README에 적는다.

### 0.0 git 규칙

- 시작 전에 `git branch --show-current`가 `feature/5th_expr`인지 확인한다. 다르면 멈추고 보고한다.
- **사용자가 지시하기 전에는 커밋하지 않는다.** 단계가 끝나면 새 파일을 `git add`까지만 하고 보고한다. push·rebase·force는 하지 않는다.
- 커밋 지시를 받으면 단계 하나당 커밋 하나를 만들고, 메시지는 `exp(E14-<단계>): <한 줄 요약>`으로 쓴다.
- 그림은 png 1장당 1MB 이하로 유지한다. 보고 직전에 `git status`와 스테이징된 파일 크기를 확인한다.

### 0.1 데이터 사실 (사전등록 때 확인함)

- `edges_long.csv.gz`: 140장(NO2\~NO12; 본시험 110장 + 재장착 30장), 이미지당 110\~111열, x≈8040\~8266, 2px 간격. 열: filename, specimen, 단계, mount, 시각, x, y_top, y_bot.
- 이 표로 다시 계산한 `d_line`이 `bearing_cross50_px`와 최대 2.6e-8px 차이로 일치한다.
- 실측치(정상 9시편) 범위는 20.009\~20.030mm다. 스케일 V1 = 0.026092 mm/px.

---

## 1. 진행 순서

| 순서 | ID | 내용 | 판정 |
|---|---|---|---|
| 1 | E14-0 | 데이터 준비 + 재현 확인 + 검정 구현 점검 + 검정력 모의실험 | 재현·점검 통과 여부 |
| 2 | E14-1 | 시편 프로파일, 창 대비, 잡음/형상 진단 (PREREG 6절) | 기술 |
| 3 | E14-2 | 후보 대표값 LOSO 비교 + 무작위화 검정 (PREREG 2\~5절) | **주 판정** |
| 4 | E14-R | 종합 보고서 | – |

- E14-1의 결과를 보고 E14-2의 후보·창 폭·검정을 바꾸지 않는다. 바꿔야 하면 PREREG 부록 B에 사후 변경으로 기록하고, 원래 결과와 나란히 낸다.
- 각 단계는 1분 안에 끝나야 한다. 무작위화 검정은 512패턴 × 비교 수만큼 LOSO를 돌리지만, LOSO 한 번이 9 fold라 계산량이 작다.

---

## E14-0. 데이터 준비와 재현 확인

- **방법** (`results/E14_0_checks.csv`, `results/E14_0_power.csv`)
  1. `edges_long`과 `values_by_image_1021`를 filename으로 붙인다(NO1 제외, 140장). 붙지 않는 행이 있으면 멈춘다.
  2. (a) 이미지마다 R0(`d_line`)를 다시 계산해 `bearing_cross50_px`와 비교한다. 완료 기준은 최대 |차이| < 0.01px다.
  3. (b) R0로 방식 3과 방식 4의 LOSO를 돌려 정상 9시편 MAE가 E6와 같은지(8.48µm, 3.99µm, 평균찍기 6.64µm) 확인한다. `calibration.py` 함수와 `run_e6_1021.py`의 시편 대표값 규칙(이미지 중앙값)을 그대로 쓴다. 차이 기준은 0.01µm다.
  4. (c) 이미지마다 W2·W3 창 안의 열 수를 세고 최소·최대를 적는다. W3 창에 20열 미만인 이미지가 있으면 보고한다. 시편 프로파일 구간(0.1mm) 중 시편의 모든 이미지에 공통으로 있는 구간 수도 적는다.
  5. (d) 검정 구현 점검: A = B = R0로 두면 T_관측 = 0이고 p = 1이어야 한다. 교환 패턴 "전부 교환"의 T가 −T_관측인지도 확인한다. 같은 함수로 방식 3 LOSO를 직접 풀어 쓴 식 e_i = (9/8)(z_i − z̄)와 비교해 차이가 1e-9µm 미만인지 본다.
  6. (e) 검정력 모의실험: PREREG 4.1절 그대로 → `E14_0_power.csv`(감소 1·3·5µm × ρ 3개 × α 2개).
- **완료 기준:** (a)·(b)·(d) 통과, (c) 표, (e) 표. 후보 R1\~R5의 **mm 오차는 아직 계산하지 않는다.**

## E14-1. 시편 프로파일과 잡음/형상 진단

- **방법:** PREREG 6절 그대로.
  - `results/E14_profile.csv`: 시편 × 구간 b(0.1mm)별 P(b) − median P, 장착별 버전 포함.
  - `results/E14_profile_contrast.csv`: 시편별 c2·c3(µm)의 중앙값·IQR, 좌·우 끝 3열 효과.
  - `results/E14_noise_shape.csv`: 시편별 σ_noise·σ_shape(µm), 장착별 값(NO7\~NO12), (R4 − R1) vs σ_noise Spearman ρ.
    - 이 진단에 R4·R1 값이 필요하므로, 여기서는 **시편 대표값만** 계산하고 실측치·오차는 보지 않는다.
  - `figures/E14_profile.png`: 시편 11개 곡선(정상은 실선, NG는 점선), 가로축 mm, 세로축 µm. W2·W3 창 경계는 세로선, 장착 내 σ 2.9µm는 회색 띠로 표시한다.
- **완료 기준:** 표 3개, 그림 1장, PREREG 6절의 미리 정한 해석 문장 1개. 이 단계에서는 실측치와 비교하지 않는다.

## E14-2. 후보 대표값 LOSO 비교

- **방법**
  1. 이미지 단위 후보(R0, R1\~R3, R2L·R3L, R1m\~R3m)를 계산한다 → `results/E14_by_image.csv`.
     - R2L·R3L은 창 안 열만으로 `robust_fit`을 다시 하고 창 중앙에서 평가한다. θ는 W1 공통값을 쓴다.
  2. 시편 대표값을 구한다. 이미지 단위 후보는 이미지 중앙값, R4·R5는 시편 프로파일에서 바로 구한다. 사진 선택 all·burst 둘 다 → `results/E14_by_specimen.csv`.
  3. 후보 × {방식 3, 방식 4} × {all, burst}마다 정상 9시편 LOSO를 돌린다 → `results/E14_loso.csv`(MAE, MAX, 시편별 err_um, NG 2시편 예측값).
  4. 무작위화 검정(PREREG 5.1절, 방식 3, all) → `results/E14_tests.csv`.
     - 주 비교 2개(R2 vs R1, R3 vs R2): T, 단측 p, Holm p, MAE 감소, "개선" 판정.
     - 기술 비교(R0 vs R1·R2L·R3L·R2·R3, R3 vs R1, R4·R5 vs R1): T와 p만 적고, 판정란에 "기술"이라고 쓴다.
- **완료 기준:** 4개 파일, PREREG 5.2절 형식의 판정 문장, E14-0 검정력 표를 인용한 해석 한 줄.

## E14-R. 종합 보고서

- `E14/README.md`는 0 결론 → 배경(멘토링) → 데이터와 사전 고정 → 단계별 표 → 사후 관찰 → 한계 → 파일 순서로 쓴다.
- 결론 맨 앞에는 주 판정 문장(R2 vs R1, R3 vs R2)을 PREREG 5.2절 그대로 쓴다. 이어서 멘토 질문(중앙 10개소 → 5개소)에 대한 답을 한두 문장으로 적고, 결과가 유의하지 않으면 "이 자료로는 구분할 수 없다"고 쓴다.
- 결론의 층위를 섞지 않는다.
  - 창 폭 효과: 주 비교
  - 파이프라인 개선: R0 비교, 기술
  - 마이크로미터 기전: 결론 없음
- 한계에는 다음을 반드시 넣는다.
  - 측정 창 밖(약 5.8mm 바깥)을 못 본 점
  - 실측 위치·각도 기록이 없는 점
  - 마이크로미터 기종 미확인
  - n=9와 검정력 표
  - 교환가능성 가정

---

## 7. 보고 형식 (단계마다)

1. 한 줄 결론
2. 핵심 수치 표(5행 이내)
3. PREREG와 다르게 한 것(없으면 "없음")
4. 그림 경로
5. 한계 한두 줄
6. 실행 명령(재현용)
7. 변경 파일 목록(`git status` 요약)
8. **Codex 리뷰에서 특히 봐 줬으면 하는 점** 1\~3개(`/codex:adversarial-review` focus text로 붙여 넣을 수 있게)

---

## 8. 리뷰가 지적하기 쉬운 부분 (미리 지킬 것)

- 통계 단위는 시편(n=9)이다. 열·이미지 단위로 p값을 내지 않는다.
- 모든 후보가 **같은 θ(W1 기울기)와 같은 xm**을 쓴다. 후보마다 기울기를 따로 맞추면 창 폭 효과와 기울기 효과가 섞인다.
- 무작위화 검정은 교환 패턴마다 **보정과 LOSO 전체를 다시 계산한다.** 시편별 |오차|의 차이를 고정해 두고 부호만 뒤집지 않는다(PREREG 부록 A).
- 교환 전에 두 후보를 각각 정상 9시편 평균으로 중심화한다. 중심화하지 않으면, 전역 오프셋 차이가 큰 후보(R4·R5)에서 교환 자체가 오차를 만든다.
- LOSO의 보정 상수 c는 fold마다 학습 시편 8개로만 맞춘다. 시편 대표값(이미지 중앙값, 시편 프로파일)은 시편 안에서만 계산하므로 누설이 없다.
- R4·R5는 탐색이다. 성능이 좋아도 마이크로미터 기전의 근거로 쓰지 않는다.
- E6 수치(방식 3 8.48µm, 방식 4 3.99µm)는 이미 봤다(PREREG 0절). R0 결과는 재현 확인과 검정력 모의실험 입력으로만 다룬다.
