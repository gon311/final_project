# E11 표면거칠기 — Claude Code 작업 지시서

> 작성 2026-10-27 · 저장소 `/Volumes/Lexar/Final/Dagyeom/git_upload/final_project` · 브랜치 `feature/3rd_expr` · 작업 폴더 `experiments/3rd/roughness/`
> 규칙 문서 `PREREG_E11.md`(같은 폴더)를 **먼저 끝까지 읽는다.** 판정 기준·지표·통계는 PREREG가 정한다. 이 문서는 구현 방법과 산출물을 정한다. 두 문서가 다르면 PREREG를 따르고, 차이를 보고한다.
> 작업 규칙·보고 형식은 `EXPERIMENTS_20261008` 0.5절(규칙 1\~14)·0.6절(저장소·커밋)·2절을 따른다. 단 브랜치는 `feature/3rd_expr`, 작업 위치는 `experiments/3rd/roughness/`로 바꿔 읽는다. 특히 규칙 3(셋업 단위 분할), 4(NG 학습 미사용), 5(시드 고정·버전 기록), 10(평균 찍기 기준선), 13(상관에는 순열검정)을 지킨다.
> **한 번에 단계 하나만 진행한다.** 끝나면 7절 형식으로 보고하고 멈춘다. 다음 단계는 사용자가 지시할 때 시작한다.
> **새 측정·재촬영은 없다.** 원본 이미지도 디코딩하지 않는다. 10/21 결과 폴더의 캐시와 CSV만 쓴다.

---

## 0. 경로와 환경

저장소 루트를 `REPO` = `/Volumes/Lexar/Final/Dagyeom/git_upload/final_project`라고 한다.

| 항목 | 경로 |
|---|---|
| 작업 폴더 | `REPO/experiments/3rd/roughness/` (이하 `E11`) |
| 규칙·지시서 | `E11/PREREG_E11.md`, `E11/EXPERIMENTS_E11_roughness_for_ClaudeCode.md`(이 문서) |
| 거칠기 원본 | `E11/data/roughness_raw.csv` (헤더 2줄 + 구분/Ra/Rq/Rz/비고, 맨 아래 측정 구간 메모 1줄) |
| 10/21 결과(커밋됨, 읽기 전용) | `REPO/experiments/3rd/E6_E10_20261021/` (이하 `R1021`) |
| 메타 | `R1021/results/meta_1021_all.csv` |
| 이미지별 측정값 | `R1021/results/values_by_image_1021.csv` |
| E6 시편별 결과 | `R1021/results/E6/calibration_by_specimen_1021.csv`, `calibration_loso_1021.csv` |
| 재장착 | `R1021/results/ratio_experiment_mounts.csv` |
| **열별 엣지 캐시(저장소 밖, 읽기 전용)** | `/Volumes/Lexar/Final/Claude outputs/20261021_3차_E6_E10/_tmp/profiles_1021.npz` (14MB, `*.npz`라 커밋하지 않음) |
| 원본 측정 코드(참고만) | `REPO/experiments/2nd/shaftlab/measure.py`(`_measure_window`, `robust_fit`, `edge_subpx`), `R1021/code/lab1021.py` |

- **`R1021`, `experiments/2nd/`, 캐시 원본은 수정하지 않는다.** 출력은 모두 `E11` 아래에 둔다. `experiments/1st/`의 미추적 폴더는 건드리지 않는다.
- 폴더 구성: `code/`, `results/`, `figures/`, `cache/`(gitignore됨, 큰 중간 파일), `README.md`(마지막 단계의 종합 보고서 = 10/21의 README와 같은 역할).
- 환경: 사용자 Mac(Apple Silicon), CPU만 쓴다. numpy·pandas·matplotlib. scipy가 없으면 Spearman·순열검정·부트스트랩을 numpy로 직접 구현한다(설치하려면 먼저 묻는다). 시드는 42로 고정하고, 패키지 버전을 README에 적는다.
- 경로는 코드에 직접 박지 말고 `code/paths.py` 한 곳에 둔다. 캐시 경로는 환경변수 `E11_PROFILES`가 있으면 그 값을, 없으면 위 기본값을 쓴다. 저장소 안 경로는 `paths.py` 위치 기준 상대경로로 만든다. 결과 CSV에는 절대경로를 쓰지 않는다.

### 0.0 git 규칙

- 시작 전에 `git branch --show-current`가 `feature/3rd_expr`인지 확인한다. 다르면 멈추고 보고한다. 새 브랜치를 만들거나 merge하지 않는다.
- **사용자가 지시하기 전에는 커밋하지 않는다.** 단계가 끝나면 새 파일을 `git add`까지만 하고 보고한다(Codex가 커밋 전 변경을 리뷰한다). push·rebase·force는 하지 않는다.
- 첫 단계(E11-0) 보고 때 `PREREG_E11.md`, 이 지시서, `data/roughness_raw.csv`도 함께 `git add`한다. PREREG가 결과보다 먼저 저장소에 들어가야 하므로, 사용자가 원하면 E11-0 전에 `docs(E11): 거칠기 실험 사전 등록` 커밋을 먼저 할 수 있다.
- 커밋 지시를 받으면 단계 하나당 커밋 하나, 메시지는 `exp(E11-<단계>): <한 줄 요약>`이다.
- `cache/`, `*.npz`, `*.parquet`(새로 생기면 `.gitignore`에 추가할지 먼저 묻는다)는 커밋하지 않는다. 그림은 png 1장당 1MB 이하로 유지한다. 보고 직전에 `git status`와 스테이징된 파일 크기를 확인한다.

### 0.1 캐시 구조 (`profiles_1021.npz`, 확인됨)

| 필드 | 모양 | 뜻 |
|---|---|---|
| `top_profiles`, `bot_profiles` | (15480, 89) float32 | 열별 세로 프로파일, 중심 ±44px |
| `top_center`, `bot_center` | (15480,) int | 프로파일 중심 y(정수) |
| `top_label`, `bot_label` | (15480,) float | 중심 대비 cross50 엣지 오프셋 |
| `x` | (15480,) int | 열의 절대 x(2px 간격) |
| `filename`, `specimen`, `bundle` | (15480,) | 이미지·시편·묶음 |
| `half_window_px`=40, `shift_buffer_px`=4, `col_step_px`=2 | | 상수 |

- 이미지 140장(본시험 NO2\~NO12 110장 + 재장착 NO7\~NO12 30장). NO1과 암막없음은 없다.
- **열별 cross50 엣지 y = `center + label`.** 이 값에 1차 강건적합을 하면 E6 `bearing_top_y`가 재현된다(NO2 1장 확인, 차이 < 0.001px).
- 본시험·재장착 구분은 `filename`으로 `meta_1021_all.csv`의 `단계`, `mount`와 붙인다.

---

## 1. 진행 순서

| 순서 | ID | 내용 | 규모 | 판정 |
|---|---|---|---|---|
| 1 | E11-0 | 데이터 준비 + 재현 확인 | 작음 | 재현 통과 여부 |
| 2 | E11-A | 시편 고유 엣지 굴곡 vs Rz | 작음\~중간 | **주 가설** |
| 3 | E11-B | 엣지 선명도 vs Ra | 작음 | 탐색 |
| 4 | E11-C | 라벨 불확도 상한 + what-if | 작음 | 계산 |
| 5 | E11-E | 오차 vs 거칠기 반증 기록 | 작음 | 확인용 |
| 6 | E11-F | 재장착 변동 vs 거칠기 | 작음 | 서술 |
| 7 | E11-R | 종합 보고서 | 작음 | – |

- E11-A를 E11-B보다 먼저 한다. A의 결과를 보고 B의 지표를 바꾸지 않는다(PREREG 4절이 이미 고정함).
- 모든 단계가 CPU로 몇 분 안에 끝나야 한다. **한 단계가 10분을 넘을 것 같으면 시작 전에 예상 시간을 보고한다.**

---

## E11-0. 데이터 준비와 재현 확인

- **방법**
  1. `code/load_roughness.py`: `roughness_raw.csv`를 읽어 `results/roughness_clean.csv`(열: rough_id, specimen_1021, Ra, Rq, Rz, Rz_Ra)를 만든다. 대응(2026-10-27 사용자 확인, 10/21과 같은 실물): NOk→NOk, NG1→NO11, NG2→NO12. `in_analysis` 열(NO1만 False)을 추가한다.
  2. `code/load_edges.py`: 캐시와 메타를 붙여 열 단위 긴 표를 만든다(열: filename, specimen, 단계, mount, 시각, x, y_top, y_bot). `cache/edges_long.csv.gz`로 저장한다(`cache/`는 커밋 안 함).
  3. 재현 확인 (`results/E11_0_checks.csv`)
     - (a) 이미지 140장 모두 `center + label`의 1차 강건적합 중심값이 `values_by_image_1021.csv`의 `bearing_top_y`·`bearing_bot_y`와 맞는지 본다. 이미지별 |차이|의 최대·중앙값을 적는다. 원 코드의 `robust_fit`(1차 → 4·1.4826·MAD 밖 제외 → 1차 재적합)을 그대로 쓴다.
     - (b) 캐시 프로파일 인덱스 i가 y = center − 44 + i에 대응하는지 본다. 프로파일에서 cross50을 다시 계산해 `label`과 비교하는데, 평활 여부를 모르므로 평활 없음과 Gaussian(1,7) σ1.2 둘 다 해 본다. 어느 쪽이 재현되는지 적는다. E11-B가 이 결과를 쓴다.
     - (c) E6 시편별 결과(`cross50`·`all`·`A_시편LOSO`)에서 평균찍기·V1·방식4의 정상 9시편 MAE가 6.64·8.48·3.99µm인지 확인한다.
     - (d) 본시험 각 시편이 10장인지, 같은 시편 10장의 x 교집합 열 수가 몇 개인지 본다.
- **완료 기준:** (a)의 최대 차이 < 0.01px(넘으면 원인을 적고 멈춤), (b) 대응 확인, (c) 재현, (d) 표. 거칠기와 이미지 지표를 **아직 비교하지 않는다.**

---

## E11-A. 시편 고유 엣지 굴곡 vs Rz (주 가설)

PREREG 3절을 그대로 구현한다. 파일: `code/e11a_edge_texture.py`.

- **방법**
  1. 본시험 10장 × 시편 11개. 이미지마다 윗엣지(주)·아랫엣지(보조) 열별 y에서 **강건 2차 추세**를 뺀 잔차를 만든다. 보조로 1차 추세 제거도 계산한다.
  2. 시편별 공통 x 격자(10장 교집합)를 만든다. 교집합이 원래 열 수의 80% 미만이면 2px 격자로 보간하고 기록한다.
  3. **Split-half 126가지 전부.** `itertools.combinations(range(10), 5)` 중 0번 이미지를 포함하는 것만 쓰면 126개가 된다. 나눔마다 A, B(5장 평균 잔차)로 `r_AB`, `S_within = cov(A,B)`(ddof=1)를 계산한다.
  4. **고정 패턴 대조.** 시편 쌍 (p, q)마다 같은 절대 x의 교집합 열에서 p의 5장 평균과 q의 5장 평균의 cov·상관을 구한다. 나눔은 시드 42로 50가지를 뽑아 중앙값을 쓴다. 교집합이 30열 미만인 쌍은 빼고 개수를 적는다.
  5. 시편별 다음을 계산해 µm(×26.092)도 함께 적는다.
     - `S_spec`, `a_spec`
     - 나눔 분포 5퍼센타일(1단계 판정용)
     - `r_AB` 중앙값
     - Spearman-Brown 보정 신뢰도 2r/(1+r)
     - `r_cross` 중앙값
  6. **판정**(PREREG 3.6절)
     - 1단계: 고유 성분 있는 시편 수 ≥ 6/11
     - 2단계: Spearman ρ(Rz, a_spec_top), 단측 순열 10000회
     - 결론 문장 셋 중 하나를 **그대로** 쓴다.
  7. **보조**(판정에 쓰지 않음)
     - 아랫엣지, 1차 추세 제거, Ra·Rz/Ra와의 ρ
     - 반복 성분 스펙트럼: 시편별 10장 평균 잔차의 파워 스펙트럼(파장 µm 축)과 고정 패턴(전 시편 평균 잔차)의 스펙트럼
  8. **재장착 대조**(NO7\~NO12, PREREG 3.5절, 서술만): lag 0과 ±10px 최대 상관을 함께 적는다.
- **산출물**
  - `results/E11A_by_specimen.csv`(시편 × {top, bot} × {poly2, poly1}: n_cols, r_AB_med, SB_rel, S_within_med, S_cross_med, S_spec, a_spec_px, a_spec_um, p05, has_specific, Rz, Ra)
  - `results/E11A_cross_pairs.csv`
  - `results/E11A_remount.csv`
  - `results/E11A_summary.csv`(판정 1·2단계 값, ρ, p, 결론 문장)
  - 그림
    - `figures/E11A_residual_examples.png`: 거칠기 상·하위 각 2시편(Rz 기준 NO10·NO3 vs NO9·NO8)의 10장 평균 잔차 프로파일 + 고정 패턴 + 두 반쪽 A/B
    - `figures/E11A_aspec_vs_Rz.png`: 산점도, 점에 시편명, ρ·p 표기
- **완료 기준:** 판정 표와 결론 한 줄, 그림 2장.
- **주의**
  - 같은 시편 10장은 독립 표본이 아니다. 통계 단위는 시편(n = 11)이다.
  - `S_spec`이 음수인 시편은 진폭 0으로 두되 원래 값도 CSV에 남긴다.
  - JPEG 8×8 블록 경계가 고정 패턴으로 보일 수 있다. 고정 패턴 스펙트럼에서 8px(208µm) 주기 성분이 보이면 보고한다.

---

## E11-B. 엣지 선명도 vs Ra (탐색)

PREREG 4절. 파일: `code/e11b_edge_sharpness.py`.

- **방법**
  1. 본시험 10장 × 11시편. E11-0 (b)에서 확인한 대응으로 프로파일 좌표를 엣지 기준 u = y − y_edge로 바꾼다.
  2. 0.1px 격자, u ∈ [−30, 30]으로 선형 보간한다. 윗엣지는 u < 0 쪽이 배경(밝음)이다. 아랫엣지는 부호를 뒤집어 같은 방향으로 맞춘다.
  3. 정규화: 엣지에서 25\~40px 떨어진 양쪽 구간의 중앙값으로 물체 0, 배경 1. 창이 ±44라서 40px까지만 쓴다.
  4. 시편별로 모든 열·10장을 평균해 ESF를 만들고 다음을 구한다.
     - 10–90% 폭(px, µm)
     - 최대 기울기
     - 시편 내 이미지별 폭의 σ(서술)
  5. Spearman ρ(Ra, 윗엣지 폭) 단측 순열 p를 낸다. 같은 폭과 촬영 순서(본시험 첫 장 시각 순위)의 ρ를 함께 낸다. 교락 표시 규칙은 PREREG 4절을 따른다.
  6. 보조: 아랫엣지, Rz와의 ρ.
- **산출물:** `results/E11B_by_specimen.csv`, `results/E11B_summary.csv`, `figures/E11B_esf_overlay.png`(시편별 ESF를 Ra 순서로 색칠), `figures/E11B_width_vs_Ra.png`
- **완료 기준:** 표와 결론 한 줄. "유의" 표현은 쓰지 않는다.

---

## E11-C. 라벨 불확도 상한과 what-if

PREREG 5절. 파일: `code/e11c_label_noise.py`.

- **방법**
  1. 입력: E6 `calibration_by_specimen_1021.csv`에서 `정의=cross50, 사진선택=all, 분할=A_시편LOSO`, 방식 `0_평균찍기`, `3_물리스케일고정(V1)`, `4_핀비율정규화선형`, 정상 9시편(NO2\~NO10).
  2. 상한: 방식4 오차 RMSE와 시편 부트스트랩 2000회 95% 구간을 구한다. `c_max = RMSE / RMS(Rz)`도 계산한다. 같은 계산을 V1·평균찍기에도 참고로 낸다.
  3. What-if: c 격자 8개 × 측정기 3종(완벽, σ_m = 5.2, σ_m = 6.4µm) × 10000회.
     - 반복마다 관측 라벨 = 참값 + ε를 만든다.
     - 평균 찍기 LOSO(학습 8시편 관측 라벨 평균)와 측정기의 MAE를 관측 라벨 기준으로 계산한다.
     - fold별로 측정기 오차가 평균 찍기보다 작은지 비교한다.
     - 규칙 ①(개선 ≥ 2.92µm), ②(≥ 7/9 fold 승), ①②를 함께 통과할 확률을 낸다.
  4. 결론 문장은 PREREG 5.2절 형식을 따른다.
- **산출물**
  - `results/E11C_upper_bound.csv`
  - `results/E11C_whatif.csv`(c × 측정기: P①, P②, P①②, MAE_측정기 중앙값, MAE_평균찍기 중앙값)
  - `figures/E11C_pass_prob_vs_c.png`: x = c(위쪽 보조축 = 평균 라벨 σ µm), 측정기별 선, c_max 세로선
- **완료 기준:** 두 표, 그림 1장, 결론 두 문장(상한 / what-if).

---

## E11-E. 오차 vs 거칠기 (반증 기록, 확인용)

PREREG 6절. 파일: `code/e11e_error_vs_roughness.py`.

- **방법**
  1. 정상 9시편. V1·방식4 오차 × {Rz, Ra}에 대해 Spearman ρ와 양측 순열 p를 낸다. 촬영 순서를 통제한 부분 Spearman도 낸다.
  2. LOSO(9 fold): 방식4 vs 방식4 + g·Rz. 학습 8시편의 시편 대표값(10/21 PREREG 5절: 모든 이미지 중앙값)으로 맞춘다. 핀 비율 입력은 `16·D/pin_px`의 시편 중앙값이다. 방식4 단독 결과가 E6 값(3.99µm)을 재현하는지 먼저 확인한다.
  3. 판정은 PREREG 6절을 따르고, "확인용(결과 일부를 사전에 봄)"을 붙인다.
- **산출물:** `results/E11E_corr.csv`, `results/E11E_loso.csv`(fold별), `results/E11E_summary.csv`
- **완료 기준:** 표와 결론 한 줄.

---

## E11-F. 재장착 변동 vs 거칠기 (서술)

PREREG 7절. 파일: `code/e11f_remount.py`.

- **방법**
  1. NO7\~NO12 장착 6점(본시험 = 10장 중앙값, 재장착 = 1장씩)으로 시편 내 σ를 구한다. 원시 cross50 px와 핀 비율 보정 둘 다 µm로 낸다.
  2. 시편 간 중앙값이 5.2 / 6.4µm를 재현하는지 확인한다.
  3. Rz·Ra와 나란히 표로 낸다. 검정은 하지 않는다.
- **산출물:** `results/E11F_remount.csv`
- **완료 기준:** 재현 여부와 표.

---

## E11-R. 종합 보고서

- `E11/README.md`를 10/21 보고서(`R1021/README.md`)와 같은 구성으로 쓴다: 0 결론 → 데이터와 사전 고정 → 단계별 표 → 사후 관찰 → 한계 → 다음 촬영 제안 → 파일.
- 결론 맨 앞에 **주 가설(E11-A) 결론 문장**을 PREREG 그대로 쓴다. 이어서 E11-C의 상한·what-if를 쓴다. 나머지는 탐색·확인용임을 표시한다.
- 한계는 PREREG 8절 목록을 빠짐없이 쓴다.
- 다음 촬영 제안은 결과에서 나온 것만 쓴다. 예를 들어 E11-C에서 라벨 잡음이 판정 기준을 막는다면 "마이크로미터 다점 측정"을 제안한다. 근거 없는 제안은 쓰지 않는다.

---

## 7. 보고 형식 (단계마다)

1. 한 줄 결론(PREREG의 결론 문장 형식)
2. 핵심 수치 표(5행 이내)
3. PREREG와 다르게 한 것(없으면 "없음"). 있으면 PREREG 부록에 "실행 중 결정" 또는 "사후 변경"으로 기록했는지
4. 그림 경로
5. 한계 한두 줄(PREREG 8절 중 해당하는 것)
6. 실행 명령(재현용)
7. 변경 파일 목록(`git status` 요약)과 5MB 넘는 파일 유무
8. **Codex 리뷰에서 특히 봐 줬으면 하는 점** 1\~3개(한 줄씩, `/codex:adversarial-review` focus text로 붙여 넣을 수 있게)

---

## 8. 리뷰가 지적하기 쉬운 부분 (미리 지킬 것)

- 통계 단위는 시편(n = 11, 오차 분석은 9)이다. 열·이미지 단위로 p값을 내지 않는다.
- E11-A의 split-half와 고정 패턴 대조를 **같은 열 격자·같은 추세 제거**로 계산한다(서로 다르면 S_spec이 편향된다).
- 결과를 보고 추세 차수, 지표, 거칠기 변수, 임계값을 바꾸지 않는다. 바꾸면 사후 변경으로 기록하고 원래 결과와 나란히 낸다.
- NG 시편(NO11·NO12)은 E11-A·B에만 넣고, mm 오차 분석(C·E)에는 넣지 않는다.
- E11-E는 결과 일부를 이미 봤다(PREREG 0절). 확인용으로만 쓴다.
