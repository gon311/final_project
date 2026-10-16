# E10. 실측 지름 파인튜닝 (멘토안 검증 + 다음 촬영 GT 학습 구조)

> `docs/EXPERIMENTS_E7resume_E10.md`(rev2 addendum) E10절. E7-close 커밋(`e16b619`) 뒤
> 시작. 완료 상태 — 8개 변형 전부 평균 찍기를 이기지 못했고, 해석 규칙(0.4절·7단계)에 따라
> "이 데이터에서는 실측 레이블 학습이 셋업 잡음을 넘지 못함"으로 결론 낸다.

## 한 줄 결론
**8개 변형 전부 평균 찍기(5.78µm)를 이기지 못했다(최선 V2_HB 19.3µm, 평균 찍기보다
13.5µm 나쁨) — 미리 고정한 해석 규칙 3개 중 1번(3.4µm 이상 개선)과 2번(10 fold 중
7승 이상)을 전부 어겨서 "이 데이터에서는 실측 레이블 학습이 셋업 잡음을 넘지 못함"으로
결론 낸다.** teacher 베이스라인(사전학습 모델 그대로, 파인튜닝 없음)조차 27.8~36.9µm로
평균 찍기보다 훨씬 나쁘다 — **파인튜닝은 teacher보다 더 나빠지는 쪽이 대부분이었다.**
V1은 H·H+B 둘 다 teacher(27.77µm)보다 오히려 나빴고(28.27~32.85µm), V2는 H가 teacher
(36.91µm)보다 나빴지만(39.66µm) **H+B만 teacher 대비 거의 절반으로 줄었다(36.91→
19.31µm)** — 그래도 평균 찍기(5.78µm)에는 한참 못 미친다. λ(teacher 유지 항)는 사실상
아무 효과가 없었다(모든 짝에서 0.01~0.2µm 차이) — px 단위 retention 항을 µm 단위
손실에 그대로 더한 설계라 스케일이 너무 작아 묻힌 것으로 보인다(아래 "결과 요약 2"
참고). **하단 단 이동 진단은 "셋업 잡음
흡수" 쪽을 가리킨다** — H 변형은 teacher·사전학습과 거의 같은 단 점유율(~95% 2단)을
유지했지만, 유일하게 단이 크게 흔들린 V2_HB는 시편마다 들쭉날쭉했다(NO10 100% 2단
유지 vs NO4 92%가 "ambiguous"로 바뀜) — "외곽 후보에 대한 간접 증거"로 읽을 근거가
없다. **NG1·NG2는 10개 변형(8+teacher 2) 중 단 하나도 올바른 방향(하한 아래)으로 NG를
잡지 못했다** — V1 계열은 NG 판정은 맞지만 방향이 반대(상한 위)였고, V2 계열은 NG1을
아예 OK로 잘못 판정했다(규칙 11).

## 멘토 제안 대응표 (0.2절, `docs/EXPERIMENTS_E7resume_E10.md`)
| 멘토 제안(2026-10-15) | 이 실험에서 한 일 | 결과 |
|---|---|---|
| 기존 모델에 dense layer 하나 추가해 학습 | E7 heatmap 헤드의 1x1 conv(`head.logits`)만 학습하는 **H** 변형 | H 단독으로는 가장 나쁜 축(V1 28.3µm, V2 39.7µm) — 헤드만 바꿔서는 지름 정보가 들어갈 자리가 거의 없다 |
| 실제 실린더 값(실측mm)을 레이블로 | 이미지 단위 **지름 손실**(Huber, µm 단위)로 top·bottom 두 네트워크를 함께 파인튜닝 | 구조는 동작했지만(코드 경로 확인 완료) 평균 찍기를 못 이겼다 |
| 답에 맞는 영역을 모델이 스스로 학습하게 | 파인튜닝 뒤 하단 엣지가 **어느 단으로 옮겨 갔는지** 진단(`step_diag.py`) | H 변형은 거의 안 옮겼고(~0.03~0.07px), 유일하게 크게 옮긴 V2_HB는 시편마다 방향이 달라 "셋업 잡음 흡수"로 판단 — "스스로 외곽을 찾았다"는 근거 없음 |
| 핀/실린더 구분 | 핀 비율 환산(V2)에서 이미지별 `pin_cross50_px`(E6 산출물)를 함께 사용 | V2는 V1보다 H에서 더 나쁘고(39.7 vs 28.3) H+B에서만 더 낫다(19.3 vs 32.8) — 일관된 우위 없음. 핀 crop 자체 학습은 범위 밖(지시서 그대로) |
| 증강으로 증폭 | 범위 밖(E2b가 따로 다룸) — 지시서가 "증강은 쓰지 않는다"로 명시 | — |

## 입력
- `cache/E7_edge1d/profiles.npz`(E7 산출물, 9347행), `E6_calibration/results/values_by_image.csv`(실측mm·`pin_cross50_px`). 둘 다 파일명으로 합친다(`data.py`).
- E7 `model.py`/`augment.py`/`train_eval.py`, E6 `calibration.py`는 **import만 하고 수정하지 않았다**(단 `train_eval.py`의 `load_cache()` 캐시 폭 검증은 지난 Codex 리뷰 반영으로 이미 커밋돼 있던 것을 그대로 재사용).

## 한 일
1. **`data.py`**: E7 캐시 + 메타 CSV를 합치는 공용 로더. `load_merged_cache(profiles_path, meta_path, pin_col, label_source)`로 경로를 인자화했고(8단계), `image_table()`로 이미지 단위 그룹(파일명→열 인덱스)과 이미지별 실측mm·핀px 표를 한 번에 만든다. `trimmed_mean_rows()`가 "이미지 단위로 묶는다"(지시서)를 열별 상·하위 10% 절사평균으로 구현한다(미분 가능, torch.Tensor 입력·출력). **LOSO 그룹(물체 ID)은 `normal_specimens(row_df)`로 데이터에서 직접 뽑는다** — `NO1`..`NO10`을 하드코딩하지 않았다. **묶음·시편·역할·실측mm은 메타 CSV에서만 읽는다**(캐시 npz의 동명 필드는 무시) — 메타가 분류 정보의 단일 출처다. `pretrain.py`/`finetune.py`/`step_diag.py`의 `main()`이 `add_io_args`/`resolve_io_args`로 `--profiles`/`--meta`/`--pin-col`/`--label-source`/`--cache-dir`/`--results-dir`를 실제로 받는다 — Codex adversarial-review 2차(2026-10-16) 전까지는 이 인자들이 저수준 함수에만 있고 CLI에는 연결돼 있지 않았다(아래 "Codex adversarial-review 반영 표" 참고).
2. **`pretrain.py`**: 2 엣지(top/bottom, heatmap raw) × (10 LOSO fold + all-normal 1) = 22회, E7과 같은 설정(200 epoch, 시드 42, E7 허용 증강)으로 teacher 복제 사전학습을 다시 했다. `E7_edge1d/checkpoints/*.pt`(전체 10시편으로 학습해 held-out을 이미 본 모델)는 시작점으로 쓰지 않았다. 체크포인트는 `cache/E10_diam_finetune/pretrain_<시편>_{top,bot}.pt`. **sanity check**: fold별 OOF median\|오차\|가 top 0.0072~0.0348px(평균 0.0142), bottom 0.0118~0.0592px(평균 0.0267)로 E7 `cv_metrics.csv`의 heatmap 범위(top 0.008~0.035, bottom/raw 0.012~0.059)와 같은 범위 — 재현 확인(`results/pretrain_sanity.csv`).
3. **`finetune.py`**: 8변형(V1/V2 환산 × H/H+B 학습범위 × λ{0,1}) × (10 fold + all-normal 1)을 파인튜닝·평가했다.
   - **지름 조립**: `d = (bot_center+bot_pred) - (top_center+top_pred)`, 이미지 단위 절사평균으로 `D_i`.
   - **환산**: V1 `pred_mm = 0.0367·D + c`, V2 `pred_mm = 10·D/pin_px + c`(묶음1 학습 제외). `c`는 학습 스칼라, 초기값은 그 fold의 teacher `D`로 구한 최소제곱(평균 잔차).
   - **학습범위**: H = heatmap 헤드(1x1 conv)만, H+B = 헤드 + 마지막 conv 블록(`conv4`/`bn4`). **BatchNorm은 두 경우 다 `eval()`로 고정**(러닝 통계가 안 흔들리게) — 지시서가 명시하지 않은 설계 선택이다.
   - **λ**: `λ·(mean|top_pred-top_pretrain_pred| + mean|bot_pred-bot_pretrain_pred|)`(px)를 Huber(µm) 손실에 그대로 더했다 — **단위를 맞추지 않고 지시서 식 그대로** 썼다(px 스케일이 µm보다 훨씬 작아 λ=1 효과가 거의 안 보이는 결과로 이어졌다, 아래 결과 참고).
   - **배치**: "이미지 단위로 묶는다"를 fold의 학습 열 전체를 한 번에 넣는 **full-batch GD**로 구현했다(fold당 학습 이미지 ~90~100장이라 미니배치가 필요 없다).
   - **중첩 LOSO 평가**: fold마다 (a) "모델 직접 예측"(변환식+c로 held-out 지름 직접 계산)과 (b) "모델 px + 기존 캘리브레이션"(같은 fold 모델로 학습 9시편의 `D_i`를 다시 계산해 E6 4방식 — 평균찍기·전체선형·물리스케일고정·핀비율선형 — 을 **fold 안에서** 새로 맞추고 held-out 예측) 둘 다 기록했다. out-of-fold px를 한꺼번에 모아 쓰지 않았다(지시서 경고 반영).
   - **teacher 베이스라인**: `run_fold(epochs=0)`로 파인튜닝 루프를 건너뛰어 "사전학습 모델 그대로"를 같은 평가 코드 경로로 돌렸다(`TEACHER_V1`/`TEACHER_V2`).
   - NG1·NG2는 all-normal 모델로 평가만 했다(규칙 4).
4. **`step_diag.py`**: 하단 프로파일마다 `shaftlab.multistep.find_peaks_subpixel`(E3과 같은 설정, n_peaks=3)로 단 위치를 구하고, teacher·사전학습 OOF·파인튜닝 OOF(8변형)가 각각 어느 단에 가장 가까운지(2px 넘으면 ambiguous) 셌다. 시편·source별 점유율을 `results/step_occupancy.csv`에 냈다.
5. **`synthetic_path_test.py`**: 다음 촬영 GT(공칭 핀게이지, `역할=gauge`) 경로가 코드로 끝까지 도는지 작은 합성 npz·CSV로 확인했다. **5개 스크립트를 전부 실제 서브프로세스 CLI로 돌린다**(`pretrain.py` → `finetune.py` → `evaluate.py` → `step_diag.py` → `make_figures.py`) — 저수준 함수를 직접 부르지 않는다(Codex adversarial-review 2차 반영, 아래 참고). 합성 npz의 `bundle`/`specimen`/`role`은 **일부러 메타와 다른 값("BOGUS", 전부 `역할=ng`)**으로 채워 뒀고, 그래도 파이프라인이 메타 쪽(`역할=gauge`)을 따라 정상 동작하는 것으로 "메타가 단일 출처"임을 이 테스트 자체가 검증한다. **핀 열이 아예 없는 메타**로도 CLI를 돌려 V2·방식4가 깨지지 않고 빠지는지, **묶음=1인 gauge 이미지**가 `apply_1008_quirks` 플래그에 따라 V2 학습에 포함/제외되는지도 추가로 확인한다(Codex adversarial-review 3차 반영, 아래 참고). 총 7개 체크. **합성 수치로 결론 내지 않는다**(지시서) — 통과/실패만 본다.
6. **`evaluate.py`**: `diam_loso.csv`/`ng_eval.csv`/`edge_shift_raw.csv`를 모아 `summary.csv`(표 첫 줄 평균 찍기, 둘째 줄부터 teacher)와 `edge_shift.csv`, 그리고 **결과를 보기 전에 고정한 해석 규칙 3개**(평균 찍기 대비 3.4µm 이상 개선 / 10 fold 중 7승 이상 / 열별 엣지 이동 중앙값 ≤1px)를 적용한 `interpretation.csv`를 만들었다.

## 결과 요약

### 1. 핵심 표 (`results/summary.csv`, `figures/mae_vs_baseline.png`)
| 변형 | MAE(µm, model_direct) | 평균 찍기 대비 | 10fold 중 승 | NG1 판정/방향 | NG2 판정/방향 |
|---|---|---|---|---|---|
| **0_평균찍기(baseline)** | **5.78** | — | — | — | — |
| TEACHER_V1(사전학습 그대로) | 27.77 | +21.99 | 1/10 | NG/상한 위(오답 방향) | NG/상한 위(오답 방향) |
| TEACHER_V2(사전학습 그대로) | 36.91 | +31.13 | 0/10 | OK(미검출) | NG/상한 위(오답 방향) |
| V1_H_l0 | 28.27 | +22.49 | 2/10 | NG/상한 위(오답 방향) | NG/상한 위(오답 방향) |
| V1_H_l1 | 28.29 | +22.51 | 2/10 | NG/상한 위(오답 방향) | NG/상한 위(오답 방향) |
| V1_HB_l0 | 32.85 | +27.07 | 1/10 | NG/상한 위(오답 방향) | NG/상한 위(오답 방향) |
| V1_HB_l1 | 32.75 | +26.97 | 1/10 | NG/상한 위(오답 방향) | NG/상한 위(오답 방향) |
| V2_H_l0 | 39.66 | +33.88 | 0/10 | OK(미검출) | NG/상한 위(오답 방향) |
| V2_H_l1 | 39.56 | +33.78 | 0/10 | OK(미검출) | NG/상한 위(오답 방향) |
| **V2_HB_l0(최선)** | **19.31** | **+13.53** | 2/10 | OK(미검출) | NG/상한 위(오답 방향) |
| V2_HB_l1 | 19.46 | +13.68 | 2/10 | OK(미검출) | NG/상한 위(오답 방향) |

**fold 승 수는 각 fold 자신의 평균 찍기 오차와 비교한 값이다**(Codex adversarial-review
[P2] 반영, 2026-10-16 4차) — baseline 오차가 fold마다 크게 다르므로(0.44~12.67µm) 전체
평균(5.78µm) 하나와 비교하면 틀린다. 아래 "Codex review 반영 표"(4차) 참고.

**실제 NG1·NG2는 둘 다 하한 아래(19.993mm, 20.002mm)인데, 10개 변형(8+teacher 2) 중
단 하나도 올바른 방향으로 잡지 못했다** — V1 계열은 "NG"는 맞혔지만 전부 반대 방향
(상한 위)이었고, V2 계열은 NG1을 아예 "OK"로 미검출했다(규칙 11, `results/ng_eval.csv`
전체는 변환·캘리브레이션 둘 다 포함해 100행).

### 2. λ(teacher 유지)는 효과가 거의 없었다
모든 짝(l0 vs l1)에서 MAE 차이가 0.01~0.2µm — H 변형(28.27→28.29, 39.66→39.56)도
H+B 변형(32.85→32.75, 19.31→19.46)도 사실상 동일하다. px 단위 retention 항을 µm
단위 Huber 손실에 그대로 더하는 설계(지시서 식 그대로, 위 "한 일 3" 참고)라, retention
값이 보통 0.1px 미만인 데 비해 diam_loss는 수십 µm대라 λ=1을 더해도 총 손실에서 차지하는
비중이 미미했던 것으로 보인다 — **이 실험에서는 "teacher 복제 유지가 지름 정확도를
해치는지"를 사실상 검증하지 못했다**(한계).

### 3. "모델 px + 기존 캘리브레이션" 경로 — 핀비율 방식만 평균 찍기에 근접
| 변형 | 0_평균찍기 | 2_전체선형 | 3_물리스케일고정 | 4_핀비율선형 |
|---|---|---|---|---|
| V1_H_l0 | 5.78 | 7.22 | 26.07 | **3.93** |
| V1_HB_l0 | 5.78 | 5.67 | 30.42 | **4.00** |
| V2_H_l0 | 5.78 | 7.18 | 28.00 | **3.95** |
| V2_HB_l0 | 5.78 | 5.42 | 47.77 | 6.81 |

H 변형(엣지를 거의 안 움직인 변형, 아래 4번)에서는 "모델 px + 핀비율선형" 경로가
평균 찍기보다도 약간 낮다(3.9~4.0 vs 5.78µm). **이걸 "파인튜닝이 개선했다"고 읽으면
안 된다** — H 변형의 엣지 이동 중앙값이 0.03~0.07px(teacher와 거의 동일, 아래 4번)이라,
이 결과는 **E6가 이미 보고한 cross50 핀비율선형 결과(`E6_calibration/results/
calibration_loso.csv`, cross50/all10/A_시편LOO, MAE 3.95µm)를 거의 그대로 재현한
것**이다(V2_H_l0 3.95µm — 소수점까지 일치). 파인튜닝이 새 정보를 더하지 않았다는 뜻이지,
파인튜닝이 효과가 있다는 근거가 아니다. H+B처럼 엣지가 많이 움직인 변형(V2_HB)에서는
이 경로도 평균 찍기보다 나빠진다(6.81µm) — 모델이 teacher에서 멀어질수록 "기존
캘리브레이션에 끼워 넣기"도 같이 나빠진다는 점에서 내적 일관성은 있다.

### 4. 열별 엣지 이동량 — H는 거의 안 움직였고, H+B(특히 V2)는 많이 움직였다 (`results/edge_shift.csv`, `figures/edge_shift.png`)
| 변형 | 상단 median(px) | 하단 median(px) | 상단 최악fold p95(px) | 하단 최악fold p95(px) |
|---|---|---|---|---|
| V1_H | 0.069 | 0.029 | 0.18 | 0.23 |
| V2_H | 0.06~0.06 | 0.03~0.04 | 0.24~0.25 | 0.19 |
| V1_HB | 0.15~0.16 | 0.20~0.23 | **2.91~2.94** | 0.76~0.79 |
| **V2_HB** | **0.68~0.69** | **0.59** | 2.22~2.26 | **3.58~3.65** |

완료 기준의 "열별 엣지 이동 중앙값 ≤1px"은 **8변형 전부 통과한다**(최댓값 V2_HB
하단 0.59px) — 하지만 **최악 fold의 95퍼센타일은 H+B 두 변형 모두 1px를 훌쩍 넘는다**:
V1_HB는 **상단**이 가장 나쁜 fold에서 최대 2.94px까지 튀고, V2_HB는 **하단**이 최대
3.65px까지 튄다 — "중앙값은 작지만(통과) 일부 fold·일부 열에서는 1px 기준을 크게
벗어난다"는 뜻이고, 어느 엣지가 불안정한지도 변형마다 다르다.

### 5. 하단 단 이동 진단 — "셋업 잡음 흡수"로 판단 (`results/step_occupancy.csv`, `figures/step_occupancy.png`)
teacher·사전학습·V1(H/HB)·V2_H는 전부 하단 단 점유율이 거의 같다(2단 ~95%, 1단 ~5%,
ambiguous ~0.1%) — 파인튜닝이 사실상 teacher가 고른 단을 그대로 따라간다. **유일하게
크게 달라진 V2_HB**(전체 2단 75.8%, 1단 13.0%, ambiguous 11.2%)도 시편별로 보면
일관성이 없다:

| 시편 | 1단 | 2단 | ambiguous |
|---|---|---|---|
| NO10 | 0.0% | 100.0% | 0.0% |
| NO8 | 1.5% | 98.5% | 0.0% |
| NO3 | 33.1% | 44.3% | 22.6% |
| **NO4** | 4.6% | 3.2% | **92.2%** |

NO10은 teacher와 완전히 같고 NO4는 92%가 ambiguous로 바뀌는 등 **셋업마다 방향과
크기가 다르다** — 판단 기준(지시서)에 따라 이건 "외곽 후보에 대한 간접 증거"가 아니라
**"셋업 잡음 흡수"**다. 멘토가 기대한 "답에 맞는 영역을 모델이 스스로 찾는다"는 신호는
이 데이터에서 보이지 않는다.

### 6. 해석 규칙 적용 결과 (`results/interpretation.csv`, 결과를 본 뒤 바꾸지 않음)
| 변형 | 1. 3.4µm 이상 개선 | 2. 7fold 이상 승 | 3. 엣지이동 ≤1px | 결론 |
|---|---|---|---|---|
| 전체 8변형 | **False**(전부 평균 찍기보다 나쁨) | **False**(최대 2/10) | True | **"이 데이터에서는 실측 레이블 학습이 셋업 잡음을 넘지 못함"** |

8변형 모두 결론이 같다 — **하나도 "실측 레이블이 정보를 담고 있을 가능성"으로 올라가지
못했다.**

## 멘토 제안과의 대응 (E6·팀원B 항목 대신)
위 "멘토 제안 대응표" 참고. 추가로: **멘토가 기대한 "범용 모델이 도메인 데이터로
특화된다"는 그림은 이 10-08 데이터(13개 셋업, 정상 10개)에서는 재현되지 않았다** —
모델이 "지름"이라는 새 신호를 봤지만, 그 신호가 셋업 간 배율 변동(최대 약 0.5%, E5·E6)
보다 작아서 학습이 디딜 곳이 없었다(0.4절에서 미리 예상한 바와 일치).

## 그림
- `figures/mae_vs_baseline.png`: 변형별 MAE(model_direct) vs 평균 찍기(위 1번).
- `figures/step_occupancy.png`: source별 하단 단 점유율(위 5번).
- `figures/edge_shift.png`: 변형별 teacher 대비 열별 엣지 이동량(위 4번).

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음**(규칙 8) — 이 결론은 "이 촬영 1회 안에서"다.
- **λ(teacher 유지)의 효과를 사실상 검증하지 못했다** — px·µm 단위를 맞추지 않고
  지시서 식을 그대로 구현한 결과, retention 항의 크기가 diam loss보다 훨씬 작아 λ=0과
  λ=1의 차이가 모든 변형에서 0.2µm 이하였다. λ가 의미 있게 작동하려면 단위를 맞추거나
  (예: retention을 µm 등가로 환산) λ 자체를 훨씬 키워야 한다 — 이번에는 지시서가 못
  박은 λ∈{0,1}을 그대로 따랐다.
- **"모델 px + 핀비율선형" 경로의 낮은 MAE(3.9~4.0µm)는 새 정보가 아니라 E6 결과의
  재현이다**(위 3번) — 보고서에 넣었지만 "파인튜닝이 효과 있다"는 근거로 쓰면 안 된다.
- **full-batch GD(미니배치 없음)** — fold당 학습 이미지가 90~100장으로 작아 택한
  단순화다. 다음 촬영에서 이미지 수가 크게 늘면(다른 구도·많은 시편) 미니배치가
  필요해질 수 있다.
- **BatchNorm을 전부 `eval()`로 고정한 것은 지시서가 정하지 않은 설계 선택이다** —
  "학습 범위(H/H+B)"가 가중치에만 적용되고 러닝 통계에는 적용되지 않는다는 뜻. 다른
  선택(H+B에서 bn4를 train 모드로 둬 배치 통계를 쓰게 하는 것)도 가능했고, 결과가
  달라질 수 있다.
- **다음 촬영 GT 경로는 코드만 확인했다**(`synthetic_path_test.py`, 5개 스크립트를 실제
  CLI로 구동 + 핀 결측·묶음1 gauge 분기 2개, 총 7체크) — 합성 데이터이므로 수치로는
  아무 결론도 못 낸다. 또한 **프로파일
  폭(half_window/shift buffer)은 E7 `augment.py`의 고정 상수(40px/4px)에 묶여 있다** —
  다음 촬영이 다른 물리적 창 폭을 쓰면(예: 50MP라 같은 mm 길이가 다른 px 수가 되는 경우)
  `data._load_profiles_npz`가 npz 메타데이터와 E7 상수가 다르다는 걸 명확한 에러로
  알리긴 하지만(Codex adversarial-review [medium] 반영), **E10 코드만으로는 처리할 수
  없다** — `A.center_crop`·`HeatmapHead`의 중심 정렬이 그 상수에 고정돼 있어서다. E7
  파일을 고치지 않는 한(지시서가 금지) 이 제약은 E10에서 풀 수 없다.
- **방식4(핀비율선형)의 "NO1 학습 제외"는 10-08 데이터(P1→P2 핀 재배치) 특유의 규칙이다**
  — `finetune.py`에 `exclude_specimens={"NO1"}`로 하드코딩돼 있다. 다음 촬영에서
  시편 이름이 다르면(또는 핀 이동 문제가 없으면) 이 규칙을 다시 봐야 한다.
- **Huber delta(10µm, 공차 반폭)는 근거가 약한 선택이다** — 지시서가 정확한 값을
  정하지 않아 공차 반폭과 같은 크기로 임의로 골랐다. 민감도 분석은 하지 않았다.

## Codex adversarial-review 반영 표 (2026-10-16, `/codex:adversarial-review --wait 보고 8번 내용`)
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] 새 촬영 입력을 실행 경로에 전달할 수 없음 — `pretrain.py`/`finetune.py`의 `main()`이 `--epochs`(·`--only-variant`)만 받고, `--profiles`/`--meta`/`--pin-col`/`--label-source`는 저수준 함수에만 있다. 합성 테스트도 `main()`을 우회해 이를 가렸다 | **맞음** — 코드로 확인. `main()`이 항상 `D.load_merged_cache()`(인자 없음)를 불렀다 | **반영함** | `data.py`에 `add_io_args`/`resolve_io_args`/`set_run_dirs`를 추가했다. `pretrain.py`/`finetune.py`/`step_diag.py`의 `main()`이 이제 `--profiles`/`--meta`/`--pin-col`/`--label-source`/`--cache-dir`/`--results-dir`를 실제로 받아 전달한다(`evaluate.py`/`make_figures.py`는 `--results-dir`(+`--figures-dir`)만). `synthetic_path_test.py`를 5개 스크립트 전부 서브프로세스 CLI로 돌리게 다시 썼다 — 실제로 돌려 통과 확인(아래 "실행 명령" 결과). |
| [high] 분류표의 물체·역할·묶음 정보가 무시되어 gauge 경로가 성립하지 않음 — `load_merged_cache`가 메타에서 `실측mm`(과 핀 열)만 가져오고, `specimen`/`bundle`/`role`은 전부 캐시 npz에서 읽는다. 합성 테스트는 이 값들을 npz에 미리 심어 결함을 가렸다 | **맞음** — 코드로 확인. 10-08 데이터는 npz의 `bundle`/`specimen`/`role`이 메타와 우연히 같은 값이라 안 드러났을 뿐이다 | **반영함** | `load_merged_cache`가 이제 `묶음`/`시편`/`역할`/`실측mm`을 **메타 CSV에서만** 읽는다(필수 열, 없으면 에러). 파일명이 메타에 없거나 메타에 중복이 있으면 조용히 넘어가지 않고 바로 에러를 낸다. 재현 확인: 10-08 데이터에서 새 코드의 `row_df["specimen"/"bundle"/"role"]`이 기존 캐시 필드와 완전히 일치(`(row_df[...] == cache[...]).all() == True`)해 **기존 결과(이미 커밋 대상인 results/*.csv)는 재계산 없이 그대로 유효**하다. 합성 테스트는 npz에 일부러 `BOGUS`/`역할=ng`를 심어 두고도 메타(`역할=gauge`)를 따라가는 것으로 검증한다. |
| [medium] 해상도 변경용 경로가 고정 89px 검증으로 차단됨 — `_load_profiles_npz`가 E7 고정 상수(`T.EXPECTED_PROFILE_WIDTH`)와 다른 폭을 무조건 거부하고, 창 폭·중심·절사 설정을 입력으로 바꿀 방법이 없다 | **부분적으로 맞음** — 검증 로직 자체는 맞지만(89px 포맷이 아니면 실제로 문제), 왜 거부하는지·무엇을 바꿔야 하는지 설명이 없었던 건 사실. 다만 `A.center_crop`·`HeatmapHead`가 E7 상수에 구조적으로 고정돼 있어 **E10 코드만으로 창 폭을 완전히 유연하게 만들 수는 없다**(E7 파일 수정 금지) | **부분 반영** | `_load_profiles_npz`가 이제 npz 자체의 `half_window_px`/`shift_buffer_px` 메타데이터로 1차 검증하고(배열-메타데이터 자기 일관성), 그 값이 E7 상수와 다르면 "E7 `augment.py`/`build_cache.py`의 상수부터 바꿔야 한다"는 구체적 에러를 낸다. **완전한 해상도 독립은 이번에 반영하지 않았다** — E7을 고치지 않는 한 불가능한 구조적 제약이라 README 한계에 명시했다(동의하지 않는 부분: "차단"을 "버그"로 보기보다 "E7-E10 경계의 구조적 한계"로 본다). |

세 지적 다 코드로 재현되는 실제 문제였다. 특히 두 번째(메타 무시)는 "다음 촬영에
GT만 바꿔 끼우면 된다"는 E10의 핵심 설계 의도를 무효화하는 결함이었다 — 10-08
데이터에서는 캐시와 메타가 우연히 일치해 드러나지 않았을 뿐이다. 세 번째는 동의하되
전부 반영하지는 않았다: E10이 E7 파일을 고치지 않기로 한 설계 제약 안에서는 완전한
해상도 독립이 불가능하므로, "왜 안 되는지"를 명확히 하는 선까지만 고쳤다.

## Codex review 반영 표 (2026-10-16, `/codex:review`)
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] `--pin-col`이 메타에 없으면 `pin_px`가 NaN으로 채워지는데, V2 학습 손실과 방식4(핀비율선형)가 둘 다 `pin_px`로 나눈다 — CLI 도움말은 "V2가 제외된다"고 하지만 실제로는 NaN이 결과에 조용히 섞여 들어간다 | **맞음** — 코드로 확인. 합성 테스트에 핀 없는 메타 시나리오가 없어서 못 잡았다 | **반영함** | `finetune.py main()`이 `row_df["pin_px"].notna().all()`을 확인해(`pin_ok`) 핀이 불완전하면 **V2 변형 전체와 모든 변형의 방식4 캘리브레이션을 명시적으로 건너뛰고** 경고를 출력한다(`run_fold`/`ng_eval_rows_for`에 `include_pin_methods` 인자 추가). `data.py`는 왜 여기서 에러를 안 내는지 문서화했다(핀 없이도 V1만 도는 게 정상 경로라서). `synthetic_path_test.py`에 핀 열 없는 메타로 실제 CLI를 돌려 V2·방식4가 빠지고 NaN이 전혀 없는지 확인하는 시나리오를 추가해 통과 확인했다. |
| [P2] 묶음1 제외가 10-08 전용 규칙인데 `label_source="gauge"`에도 무조건 적용됨 — 합성 메타가 묶음1을 썼는데도 V1만 테스트해서 이 경로가 검증되지 않았다 | **맞음** — 코드로 확인 | **반영함** | `run_fold`/`ng_eval_rows_for`에 `apply_1008_quirks` 인자를 추가해 `main()`이 `label_source=="micrometer"`일 때만 `True`를 넘긴다 — 묶음1 V2 제외와 방식4의 "NO1" 제외 둘 다 이 플래그로 묶었다(같은 종류의 10-08 전용 매직이라 함께 고쳤다, 지적 범위보다 넓게 반영). `synthetic_path_test.py`가 `run_fold`를 `apply_1008_quirks=True/False`로 직접 불러 묶음=1 gauge 이미지가 각각 제외/포함되는지 비교해 통과 확인했다. |

10-08 데이터(기본 `label_source="micrometer"`)에서는 `pin_ok=True`, `apply_1008_quirks=True`로
그대로 유지돼 기존 동작·기존 결과가 바뀌지 않는다(재확인함) — 두 수정 다 **다음 촬영
(`label_source="gauge"`) 경로에만 영향을 준다.

## Codex review 반영 표 2 (2026-10-16, 4차, `/codex:review`)
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] `fold_win_count`가 각 fold를 **전체 평균** baseline MAE(5.78µm)와 비교하는데, baseline 오차는 fold마다 크게 다르다(0.44~12.67µm) — `V1_H_l0`가 보고된 0승이 아니라 실제로는 fold별로 짝지으면 2승이다 | **맞음** — 코드로 확인하고 재계산해 정확히 재현됨(`V1_H_l0` 0→2승) | **반영함** | `evaluate.py`에 `baseline_by_fold(diam)`를 추가해 **같은 fold의 '0_평균찍기' 오차와만** 비교하도록 `summarize_variant`를 고쳤다. `results/summary.csv`·`interpretation.csv`를 재계산(재학습 불필요, `evaluate.py`만 재실행)해 커밋 대상 수치를 갱신했다(아래 "결과 요약 1" 표). **결론(해석 규칙 적용 결과)은 바뀌지 않는다** — 고친 뒤에도 모든 변형이 7/10 기준(조건 2)에 한참 못 미친다(최대 2/10). |

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate
cd E10_diam_finetune
python3 pretrain.py          # cache/E10_diam_finetune/pretrain_*.pt, pretrain_oof.npz(커밋 안 함)
                              # results/pretrain_sanity.csv — 22회 학습, 200 epoch, 약 24분(MPS)
python3 finetune.py          # results/diam_loso.csv, edge_shift_raw.csv, ng_eval.csv
                              # cache/E10_diam_finetune/finetune_oof.npz(커밋 안 함)
                              # — 8변형 x 11회 파인튜닝(50 epoch) + teacher 베이스라인, 약 16분(MPS)
python3 evaluate.py          # results/summary.csv, edge_shift.csv, interpretation.csv
python3 step_diag.py         # results/step_occupancy.csv
python3 make_figures.py      # figures/*.png
python3 synthetic_path_test.py  # 다음 촬영 GT 경로를 5개 스크립트 전부 실제 CLI로 구동 +
                                 # 핀 열 없는 메타, 묶음1 gauge 분기 2개 추가 확인(총 7체크)
                                 # (서브프로세스, 통과/실패만, 수 초) — 재실행해 통과 확인함
```

다음 촬영(공칭 핀게이지 GT)이 오면 모든 스크립트에 아래 공통 인자를 붙인다:
```bash
--profiles <새 profiles.npz> --meta <분류표.csv> --pin-col <핀px 열 이름> \
--label-source gauge --cache-dir cache/E10_next --results-dir E10_diam_finetune/results_next
```

### 환경·패키지 버전
E7_edge1d/README.md와 동일(Python 3.11.9, torch 2.14.1, pandas 3.0.6 등) — 새로
설치한 패키지 없음. `device = "mps" if torch.backends.mps.is_available() else "cpu"`,
이 Mac은 MPS 사용. 랜덤 시드: 사전학습 42(`T.set_seed`), 파인튜닝 42
(`torch.manual_seed`, `run_fold` 진입 시 고정).

## 변경 파일 목록
```
A experiments/2nd/E10_diam_finetune/README.md
A experiments/2nd/E10_diam_finetune/data.py
A experiments/2nd/E10_diam_finetune/pretrain.py
A experiments/2nd/E10_diam_finetune/finetune.py
A experiments/2nd/E10_diam_finetune/evaluate.py
A experiments/2nd/E10_diam_finetune/step_diag.py
A experiments/2nd/E10_diam_finetune/make_figures.py
A experiments/2nd/E10_diam_finetune/synthetic_path_test.py
A experiments/2nd/E10_diam_finetune/results/diam_loso.csv
A experiments/2nd/E10_diam_finetune/results/summary.csv
A experiments/2nd/E10_diam_finetune/results/interpretation.csv
A experiments/2nd/E10_diam_finetune/results/edge_shift.csv
A experiments/2nd/E10_diam_finetune/results/edge_shift_raw.csv
A experiments/2nd/E10_diam_finetune/results/step_occupancy.csv
A experiments/2nd/E10_diam_finetune/results/ng_eval.csv
A experiments/2nd/E10_diam_finetune/results/pretrain_sanity.csv
A experiments/2nd/E10_diam_finetune/figures/mae_vs_baseline.png
A experiments/2nd/E10_diam_finetune/figures/step_occupancy.png
A experiments/2nd/E10_diam_finetune/figures/edge_shift.png
```
`cache/E10_diam_finetune/`(체크포인트 110+22개, `pretrain_oof.npz`, `finetune_oof.npz`,
총 약 3.8MB)는 `.gitignore`(`**/cache/`, `*.pt`)에 의해 커밋되지 않는다 — 스크립트
재실행으로 재생성된다(시드 고정, 사전학습 체크포인트는 파일이 있으면 건너뛴다). 5MB
초과 파일 없음.

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- λ(teacher 유지 항)를 px 단위 그대로 µm 단위 Huber 손실에 더한 설계(위 한계 2번)가
  "λ 효과가 없다"는 결론 자체를 무의미하게 만드는 건 아닌지 — 단위를 맞추지 않은 게
  구현 결함인지, 지시서 식을 문자 그대로 따른 결과로 봐도 되는지.
- "모델 px + 핀비율선형" 경로가 평균 찍기에 근접한 것(3.9~4.0µm)을 "E6 재현일 뿐 새
  정보 아님"으로 해석한 논리(결과 요약 3번)가 맞는지 — 엣지 이동이 작다는 근거(0.03~
  0.07px)만으로 그렇게 단정해도 되는지, 아니면 별도로 확인할 방법이 있는지.
- `trimmed_mean_rows`(이미지 열 상·하위 10% 절사평균)가 미분 가능한 형태로 구현됐는데,
  `torch.sort`를 거치는 절사 과정이 그래디언트를 왜곡하지 않는지(선택된 열만 그래디언트를
  받고 잘린 열은 받지 않는 구조가 의도한 대로인지) 봐 달라.
- BatchNorm을 H/H+B 둘 다 `eval()`로 고정한 설계 선택(위 한계 참고)이 "학습 범위"
  비교의 공정성을 해치지 않는지 — 특히 H+B에서 `bn4`도 학습 대상에 넣었는데 러닝
  통계는 안 바꾼 게 일관성 있는 선택인지.
- NG1·NG2 방향 전부 틀린 것(결과 요약 1번)이 "모델이 셋업 잡음에 끌려가서 생긴 우연"
  인지, 아니면 구조적으로 설명되는 패턴(예: c의 부호나 V1/V2 변환식 자체의 편향)인지.
