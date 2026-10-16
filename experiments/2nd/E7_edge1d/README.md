# E7. 1D 엣지 프로파일 학습 파이프라인 프로토타입

> **진행 상태: 1단계 완료(2026-10-15).** teacher(cross50) 복제 확인. 다음 단계는
> E10(실측 지름 파인튜닝)이다. 멈췄던 이유는 정답(GT) 데이터가 없어 "teacher를 얼마나
> 잘 복제하는가"까지만 확인할 수 있었기 때문이다. 재개 방향은 멘토 피드백(2026-10-15)과
> `docs/EXPERIMENTS_E7resume_E10.md` 0.3절(지름 손실로 학습하는 구조가 GT가 와도
> 그대로 필요하다는 논거) — 아래 "다음 단계" 참고.

## 한 줄 결론
**heatmap(soft-argmax) 헤드는 상단·하단 둘 다 완료 기준(teacher 대비 median |오차|
≤0.05px)을 통과하고(0.014~0.027px), 회귀 헤드는 둘 다 통과하지 못한다(0.088~0.177px)
— 헤드 선택이 지금은 "상단이냐 하단이냐"보다 더 결정적이다.** 처음 버전은 창 중심을
`round(teacher)`로 잡아 모델이 "teacher의 반올림 오차(±0.5px)만 보정"하는 순환 과제를
풀었다는 Codex adversarial-review [high] 지적을 받고, 중심을 teacher와 무관한
`coarse_section`의 1/4 해상도 좌표로 바꿨다(새 라벨 범위: 상단 -5~0px, 하단 +5~+15px) —
그 결과 과제가 훨씬 어려워져 작은 3층 모델(5천 파라미터)로는 0.09~0.19px까지 밖에
못 갔고, 4층 34.5천 파라미터로 키우고서야 heatmap 헤드가 기준을 다시 통과했다. **shift
−0.5px 방향만 유난히 불안정했던 이전 결과는 81px로 잘라낸 프로파일에 스플라인 외삽을
쓴 아티팩트였다**(Codex [medium] 지적, 코드로 재현) — 프로파일을 89px(모델 입력+버퍼
4px씩)로 저장하고 버퍼 안에서만 보간하도록 고치자 heatmap 헤드의 shift 방향별 차이가
거의 사라졌다. **셋업 내 σ는 처음으로 teacher를 평균·최악 둘 다에서 넘어서는 조합이
나왔다**(상단·하단 둘 다 heatmap 예측, σ=0.1275 vs teacher 0.1357) — 다만 그 차이
(0.008px ≈ 0.3µm)는 E6가 잰 반복성 노이즈 바닥(0.093px ≈ 3.4µm)보다 훨씬 작아 "확실히
이겼다"고 주장할 근거는 아직 아니다.

## 한 일
1. `shaftlab/measure.py`의 `extract_profiles`에 두 옵션을 추가했다(둘 다 기본값은 기존
   동작 유지, `E1_module/reproduce.py` 재실행으로 영향 없음 확인):
   - `return_labels=True`: `(profiles, labels, xs, centers)`를 반환.
   - **`coarse_seg`(Codex adversarial-review [high] 반영)**: `coarse_section()`의
     원본(1/4 해상도) 반환값을 넘기면, 프로파일 창 중심을 teacher(cross50)가 아니라
     `_measure_window`가 `edge_subpx`를 부르기 **전** 쓰는 coarse 좌표로 잡는다. 기본값
     (`None`)이면 이전처럼 `round(teacher)`를 중심으로 쓴다(순환 설계, 아래 "한 일 — 이전
     버전과의 차이" 참고) — 이 옵션 없이는 라벨이 ±0.5px 반올림 오차뿐이라 모델이 "엣지를
     찾는" 과제를 풀지 않는다.
2. `build_cache.py`: `coarse_section` + `measure_bearing`(cross50, E1 그대로)으로 section·
   coarse seg를 구한 뒤 `extract_profiles(..., return_labels=True, coarse_seg=seg)`로
   프로파일·라벨·중심을 뽑는다. **프로파일을 89px로 저장한다**(모델이 쓰는 81px +
   `SHIFT_BUFFER_PX=4`씩 — Codex [medium] 반영, shift 증강이 외삽 없이 돌게 하려는
   목적, 아래 3번). `half_window_px=40`(규칙 14), `col_step_px=2`(서브샘플링 없음) —
   9347행(상·하단 프로파일 쌍)을 26초에 만든다.
3. `augment.py`: **입력 계약이 바뀌었다** — 모든 함수가 89px 넓은 프로파일을 받고 81px
   크롭을 반환한다. `shift1d`는 이 4px 버퍼 안에서만 보간한다(±0.5px 이동에 필요한 여유는
   0.5px뿐이라 버퍼가 넉넉하다 — 외삽 없음을 수치로 확인, 아래 "Codex adversarial-review
   반영 표"). **블러 허용 목록도 다시 검토했다(Codex [medium] 반영)** — E2 표의 "셋업최악"
   기준까지 보면 **상단은 E2가 테스트한 블러 세 강도(σ=0.5/1.0/2.0) 전부 게이트**라
   상단 블러를 완전히 뺐다. 하단은 전체중앙값·셋업최악 둘 다 "허용"인 σ∈{0.5,1.0}만
   남겼다(σ=0.25는 E2가 테스트하지 않은 값이라 애초에 근거가 없었다 — 처음 버전의 또 다른
   실수). brightness(상단 {-10,-20}, 하단 {-10,-20,-40}, 어둡게만), contrast(×0.8),
   subpixel shift(±0.5px, 라벨이 정확히 함께 이동)는 그대로다. **여전히 "비대칭 블러·링잉
   세기·내부 반사 레벨·가우시안 노이즈·회전"은 뺀다**(E2 미검증/2D 연산이라 1D 프로파일에
   직접 적용 불가, 한계 참고).
4. `model.py`: `EdgeCNN1D` — bias 없는 Conv1d 4개(c1=32,c2=48,c3=48,c4=48) + BatchNorm
   백본(≈34.5천 파라미터, 50k 이하) 위에 회귀 헤드(전역 평균 풀링+Linear) 또는 heatmap
   헤드(1x1 conv 로짓+softmax+soft-argmax)를 올린다. **처음 버전(3층, 5065개)은 새 라벨
   (teacher-무관 coarse 중심)로 돌리니 300 epoch까지 늘려도 0.141px(기준 미달)에서
   수렴이 느려졌다** — 용량을 키우고서야 150 epoch에 0.094px, 250 epoch에 0.075px로
   계속 내려갔다(결과 요약 1번 참고). 입력은 프로파일마다 중앙값·표준편차로 z-score
   정규화한다 — **이 설계 덕분에 brightness·contrast 증강에서 모델 예측이 수학적으로
   완전히 불변이다**(실측 확인: 밝기 -10/-20/-40, 대비 0.8 전부 clean과 소수점까지 같은
   mae_px).
5. `train_eval.py`: `specimen` 필드(0.3절 분류표가 NO1=묶음1+13을 이미 합쳐 줌) 기준
   `LeaveOneGroupOut` — 정상 10개 시편 LOSO(규칙 3). NG1·NG2는 학습·검증 fold에 전혀 안
   쓴다(규칙 4) — all-normal 체크포인트에만 평가용으로 한 번 돌린다. 학습 배치마다
   `augment.py`의 허용 증강(또는 identity) 중 하나를 무작위로 적용한다. Huber loss,
   Adam(lr=1e-3) + CosineAnnealingLR, **200 epoch**(아래 "결과 요약 1" 수렴 추이 보고
   정함), 시드 42. 베이스라인(규칙 10 취지 번역) = "오프셋 항상 0"(정수 중심 그대로).
   상단·하단(raw/flipped) × (회귀/heatmap) 6개 조합. **외경 재계산 σ**: 같은 행(이미지의
   같은 열)에 상단·하단 라벨이 짝지어 있으므로 `diam=(bot_center+bot_label)-
   (top_center+top_label)`로 조립한다 — 상단만 예측으로 바꾸고 하단은 teacher 그대로(Stage
   A), 상단·하단 둘 다 예측(Stage B, heatmap·회귀 둘 다 — "정확도로 해석하지 않음"이라고
   명시). oof 예측은 `cache/E7_edge1d/oof_predictions.npz`에 따로 저장해(커밋 안 함)
   재학습 없이 후처리만 바꿀 수 있게 했다.
6. `make_figures.py`: 그림 3장(아래 "그림") — clean MAE 그림은 baseline과 모델의 차이가
   100배 넘게 벌어져 로그축으로 바꿨다.

## 결과 요약

### 1. 모델 용량·epoch 수렴 추이 (튜닝 과정, `top/회귀` 기준)
| 모델 | epoch | median \|오차\|(px) |
|---|---|---|
| 3층(5065 파라미터) | 120 | 0.187 |
| 3층(5065 파라미터) | 300 | 0.141 |
| 4층(34481 파라미터) | 150 | 0.094 |
| 4층(34481 파라미터) | 250 | 0.075 |
| 4층(34481 파라미터, **최종**) | 200 | 0.088 |

**300 epoch까지 늘려도 작은 모델은 0.14px 아래로 못 내려갔다** — 용량 부족이 epoch
부족보다 더 큰 병목이었다는 뜻이다. 4층으로 키우자 150→250 epoch 사이에도 계속
떨어져(0.094→0.075) 완전히 수렴하지 않았을 가능성이 있다 — 시간 예산상 200 epoch에서
멈췄다(한계 참고). **이 표는 회귀 헤드 기준이다 — heatmap 헤드는 같은 200 epoch에서
이미 0.014px로 훨씬 낮다**(아래 2번), 즉 헤드 구조 자체가 용량·epoch보다 더 큰 차이를
만든다.

### 2. Clean teacher 오차 (`results/cv_metrics.csv`, `figures/clean_mae_by_combo.png`)
| 모델 | median \|오차\|(px, LOSO fold 평균) | fold 최악 median(px) | 실제 max(px) | 완료 기준(≤0.05px) |
|---|---|---|---|---|
| (베이스라인) top, 오프셋 항상 0 | 1.728 | 3.279 | 5.005 | — |
| (베이스라인) bottom, 오프셋 항상 0 | 8.643 | 11.636 | 14.770 | — |
| top / 회귀 | 0.088 | 0.169 | 1.738 | **미달** |
| top / heatmap | **0.014** | 0.035 | 0.236 | 통과 |
| bottom / 회귀 / raw | 0.144 | 0.352 | 0.964 | **미달** |
| bottom / 회귀 / flipped | 0.177 | 0.498 | 1.097 | **미달** |
| bottom / heatmap / raw | 0.027 | 0.059 | 0.363 | 통과 |
| bottom / heatmap / flipped | **0.022** | 0.043 | 0.212 | 통과 |

**Codex adversarial-review [P2] 반영**(2026-10-16): 이전에 "max(px)"로 적은 값은 실제
최대 오차가 아니라 **11 fold 각각의 median \|오차\|(=`mae_px`) 중 최악값**이었다(예:
`top / heatmap`의 0.035는 fold-median의 최댓값이고, `cv_metrics.csv`의 진짜
`max_abs_err_px`는 0.236이다). 열 이름을 "fold 최악 median(px)"으로 바로잡고, 전체
fold에 걸친 진짜 최댓값(`max_abs_err_px`의 최댓값)을 "실제 max(px)" 열로 추가했다.
완료 기준(≤0.05px)은 애초에 median 기준이라 판정은 바뀌지 않지만, 꼬리 오차는
"fold 최악 median"보다 2~6배 크다(특히 top/회귀 0.169→1.738, bottom/회귀/flipped
0.498→1.097) — "결과 요약 3·4"(증강 강건성, 셋업 내 σ)의 median 기반 해석은 그대로
유효하지만, "개별 열 하나가 최대 얼마나 틀릴 수 있는가"를 볼 때는 이 "실제 max(px)"
열을 봐야 한다.

- **헤드 선택이 핵심 변수다.** heatmap은 베이스라인보다 60~120배 낮고 완료 기준을
  여유 있게 통과한다. 회귀는 베이스라인보다는 훨씬 낫지만(상단 20배, 하단 50~60배)
  완료 기준엔 못 미친다 — 전역 평균 풀링으로 압축한 뒤 바로 스칼라를 회귀하는 구조가
  이 넓어진(상단 5px, 하단 10px) 탐색 범위에서 위치 정보를 충분히 보존하지 못하는
  것으로 보인다(soft-argmax는 공간적 위치 자체를 기대값으로 직접 뽑아 더 유리한 구조적
  가정을 갖는다).
- **좌우 반전(flipped)의 효과가 헤드마다 반대다** — heatmap은 flipped가 raw보다 조금
  낫고(0.027→0.022), 회귀는 flipped가 오히려 더 나쁘다(0.144→0.177). 처음 버전(순환
  라벨 설계)에서는 "flipped가 둘 다 낫다"였는데, 지금은 일관되지 않는다 — 이전 가설
  (상단과 방향을 맞추면 항상 더 쉽다)은 기각한다.

### 3. 증강 강도별 강건성 (`results/cv_metrics.csv`, `figures/robustness_by_aug.png`)
**shift 방향 비대칭이 해소됐다.** heatmap 헤드 기준:
| 엣지(raw) | shift −0.5px | shift +0.25px | shift +0.5px |
|---|---|---|---|
| top | 0.0195 | 0.0151 | 0.0160 |
| bottom | 0.0269 | 0.0270 | 0.0280 |

이전 버전(81px 크롭에 외삽 shift)은 top heatmap에서 -0.5px=0.061 vs +0.25px=0.026(약
2.3배 차이)였다 — 지금은 0.0195 vs 0.0151(약 1.3배)로 거의 평평하다. **brightness·
contrast는 heatmap·회귀 둘 다 clean과 소수점까지 완전히 같은 값이다**(z-score 정규화
덕분에 수학적으로 불변 — 위 "한 일 4" 참고). **블러(하단만, σ=0.5/1.0)도 clean 대비
거의 안 움직인다**(heatmap raw: 0.0267→0.0253~0.0285). 전체적으로 **heatmap 헤드는
모든 증강·강도에서 0.03px 안쪽에 머문다** — 완료 기준(0.05px)에 여유가 있다.

### 4. 셋업 내 지름 σ — teacher vs 모델 재계산 (`results/setup_sigma.csv`, `figures/setup_sigma.png`)
| 모델 | 셋업 평균 σ(px) | 셋업 최악 σ(px) | teacher 대비 |
|---|---|---|---|
| **stageB_both_heatmap_raw(탐색, 정확도 아님)** | **0.1275** | **0.2977** | 평균 -0.0082, 최악 -0.0474 (**둘 다 더 낮음**) |
| stageA_bottom_heatmap_raw(상단=teacher) | 0.1278 | 0.2990 | 평균 -0.0079, 최악 -0.0461 (**둘 다 더 낮음**) |
| stageA_bottom_reg_raw(상단=teacher) | 0.1305 | 0.2790 | 평균 -0.0052, 최악 -0.0661 |
| stageA_bottom_heatmap_flipped(상단=teacher) | 0.1316 | 0.3120 | 평균 -0.0041, 최악 -0.0331 |
| stageA_top_heatmap(하단=teacher) | 0.1356 | 0.3425 | 평균 -0.0001(동률), 최악 -0.0026 |
| **teacher(cross50 전체)** | **0.1357** | 0.3451 | — |
| stageA_bottom_reg_flipped(상단=teacher) | 0.1361 | 0.2869 | 평균 +0.0004(동률), 최악 더 나음 |
| stageA_top_reg(하단=teacher) | 0.1528 | 0.6189 | 평균 +0.0171, 최악 +0.2738 (뚜렷이 나쁨) |
| stageB_both_reg_raw(탐색, 정확도 아님) | 0.1576 | 0.5657 | 평균 +0.0219, 최악 +0.2206 (뚜렷이 나쁨) |

**처음으로 teacher를 평균·최악 둘 다에서 넘어서는 조합이 나왔다**(상단·하단 둘 다
heatmap 예측 — Stage B, 그리고 하단만 heatmap 예측한 Stage A) — 하지만 그 차이
(최대 0.0082px ≈ 0.3µm)는 **E6의 반복성 노이즈 바닥(cross50 σ≈0.093px=3.4µm,
`E6_calibration/results/within_mount_repeatability.csv`)보다 30배 가까이 작다** —
"모델이 teacher보다 측정을 개선했다"고 주장할 근거는 아니지만, 적어도 "teacher를 거의
완벽히 복제하면서 나쁘지 않은 쪽으로 치우쳤다"는 더 분명한 신호다. **회귀 헤드 조합은
뚜렷이 나쁘다**(top/회귀 +0.017px, Stage B 회귀 +0.022px) — clean MAE가 나쁜 모델은
지름 재계산 σ도 같이 나빠진다는, 직관과 일치하는 결과다.
- **주의할 methodological 포인트:** `stageA_bottom_reg_raw`는 clean MAE가 나쁜데도
  (0.144px, 미달) σ는 teacher보다 살짝 낫다(0.1305 vs 0.1357) — σ는 "셋업 내 10장
  사이 일관성"(정밀도)이지 "teacher와 얼마나 가까운가"(정확도)가 아니기 때문에, 같은
  방향으로 치우친 편향(bias)이 안정적이면 정확도가 나빠도 정밀도는 괜찮을 수 있다.
  두 지표를 같이 봐야 하는 이유다.

### 5. NG1·NG2 평가 (학습에 전혀 안 씀, 규칙 4)
| 엣지/모델 | mae_px | max_abs_err_px |
|---|---|---|
| top/회귀 | 0.053 | 0.769 |
| top/heatmap | **0.008** | 0.302 |
| bottom/회귀/raw | 0.125 | 1.270 |
| bottom/회귀/flipped | 0.110 | 1.345 |
| bottom/heatmap/raw | **0.016** | 0.113 |
| bottom/heatmap/flipped | **0.016** | 0.096 |

heatmap 헤드는 NG(10개 정상 시편으로 학습, 전혀 못 본 분포밖 시편)에서도 median 오차가
오히려 정상 시편 clean보다 낮다(특히 top: 0.008 vs 0.014) — coarse 중심에서 실제 엣지를
찾는 과제로 바뀐 뒤에도 일반화가 잘 된다는 뜻이다. 회귀 헤드는 NG에서 더 나쁘고 꼬리
(max_abs_err)가 1px를 넘는 경우도 있다 — 애초에 clean에서도 약한 모델이라 일관된다.

## 팀원B 결과와 대조
팀원B는 이 실험(1D 프로파일 CNN 학습)을 직접 하지 않았다. 대신 `1012_edgecheck`의 결론
("하단 경계는 사진 안에 정답이 없고, 가진 정답은 시편 12개의 지름뿐이라 지금은 학습으로도
못 간다")을 teacher 해석의 근거로 그대로 인용한다 — **이 실험의 결과(heatmap은 teacher를
거의 완벽히 복제하지만 σ 개선은 반복성 노이즈 바닥보다 작다)는 그 결론과 같은 방향이다**:
CNN이 "teacher를 잘 따라 하는 것"과 "측정이 더 좋아지는 것"은 여전히 다른 얘기이고,
팀원B가 말한 "사진 안에 정답이 없다"는 근본 한계를 이 실험이 해소하지는 못했다.

## 그림
- `figures/clean_mae_by_combo.png`: clean 상태 teacher 오차, 모델/엣지/변형별(로그축) +
  0.05px 기준선.
- `figures/robustness_by_aug.png`: 증강 강도별 강건성(상단/하단, 회귀 vs heatmap).
- `figures/setup_sigma.png`: 셋업 내 지름 σ, teacher vs 모델 재계산 조합별.

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음(규칙 8)** — 이 결론은 "이 촬영 1회 안에서 teacher를 얼마나
  잘 복제하는가"다. 최종 성능 주장이 아니다.
- **teacher 자체가 GT가 아니다.** E4가 cross50을 1순위로 고른 근거는 "현재 촬영 조건 안에서의
  반복성"이지 물리적 정확성이 아니다. 창 중심을 teacher-무관으로 바꿨어도(이번 수정) 라벨
  자체(= 맞혀야 할 정답)는 여전히 cross50이다 — "teacher를 얼마나 잘 흉내 내는가"이지
  측정 정확도가 아니라는 지시서 요구사항은 그대로 유효하다(팀원B `1012_edgecheck` 인용,
  위 "팀원B 결과와 대조").
- **4층 34.5천 파라미터·200 epoch는 수렴 추이(위 "결과 요약 1")에서 아직 완전히 평평해지지
  않은 지점에서 멈췄다** — 150→250 epoch 사이 회귀 헤드가 계속 좋아지고 있었다(0.094→
  0.075). 더 긴 학습이나 더 큰 용량으로 회귀 헤드도 기준을 통과할 수 있는지는 확인하지
  못했다(시간 예산 제약, 6개 조합 × 200 epoch가 이미 약 75분 걸렸다).
- **지시서가 적어 둔 증강(비대칭 블러, 링잉 세기, 내부 반사 레벨, 가우시안 노이즈)을 전부
  뺐다** — E2가 이 네 종류를 테스트하지 않아 "E2 허용" 필터를 통과하지 못한다(1차 Codex
  리뷰에서도 물었던 점, 아직 미해결).
- **회전 증강은 1D 프로파일 단계에서 아예 뺐다** — 2D 기하 연산이라 이미 추출된 프로파일에는
  재현할 수 없다. 넣으려면 "원본 이미지에 회전 적용 → 프로파일 다시 추출"을 증강 파이프라인
  안에 넣어야 한다.
- **"셋업 내 σ가 teacher보다 같거나 작다"는 완료 기준은 방향은 맞지만(최선 조합 2개가 평균·
  최악 둘 다 낮음) 반복성 노이즈 바닥보다 작은 차이라 "확실히 이겼다"고 말할 근거는 없다.**
- **라벨만 바꾸면 GT로 재학습할 수 있는 구조는 그대로 유지했다** — `build_cache.py`가 라벨과
  프로파일을 분리해 저장하므로, 다음 촬영에서 핀게이지 세트 기반 GT가 생기면 그 라벨로
  `profiles.npz`의 `*_label` 필드만 바꿔 `train_eval.py`를 그대로 재사용할 수 있다.
- **`coarse_section`의 1/4 해상도 양자화 자체가 "teacher-무관"이라고 완전히 보장하지는
  않는다** — `coarse_section`과 `edge_subpx`(teacher 계산)가 같은 원본 이미지·같은
  대략적 밝기 문턱 로직을 공유하므로, 극단적으로 같은 종류의 오차(예: 특정 조명 조건에서
  둘 다 같은 방향으로 치우치는 경우)가 완전히 없다고는 못 한다 — "teacher와 독립"은
  "teacher의 정확한 서브픽셀 계산과는 별개의, 훨씬 거친 경로를 거친다"는 뜻이지 통계적
  독립성을 증명한 건 아니다.

## 다음 단계
- **회귀 헤드는 완료 기준 미달이라(top 0.088px, bottom 0.144~0.177px, 기준 ≤0.05px)
  이후 단계(E10)에서는 heatmap 헤드만 쓴다.**
- "Codex 리뷰에서 특히 봐 줬으면 하는 점" 중 회귀 헤드 관련 두 항목(원인 분석 — 구조
  한계 vs 용량·epoch 부족, shift 방향별 불안정)은 더 조사하지 않는다 — 회귀 헤드를 더
  쓰지 않기로 했으므로 원인을 더 파는 것이 E10에 필요하지 않다.
- `coarse_seg` 기반 중심이 teacher와 "충분히" 독립적인지는 미해결 상태로, E10에도 그대로
  이어지는 한계로 남긴다(위 "한계" 마지막 항목 — `coarse_section`과 `edge_subpx`가 같은
  원본 이미지·같은 대략적 밝기 로직을 공유해 통계적 독립성을 완전히 보장하지 않는다).

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate   # 없으면: python3.11 -m venv .venv && pip install -r requirements.txt
cd E7_edge1d
python3 build_cache.py       # cache/E7_edge1d/profiles.npz (약 26초, 커밋 안 함)
python3 train_eval.py        # results/cv_metrics.csv, results/setup_sigma.csv,
                              # checkpoints/*.pt·cache/E7_edge1d/oof_predictions.npz(둘 다 커밋 안 함)
                              # — 6개 조합 x 11회 학습, 200 epoch, 약 75분(MPS)
python3 make_figures.py      # figures/*.png

# E1 영향 재확인(shaftlab/measure.py의 extract_profiles를 수정했다)
cd ../E1_module && python3 reproduce.py
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0,
  matplotlib 3.11.2, scipy 1.17.1
- torch 2.14.1, scikit-learn 1.9.1 (신규 설치, 설치 전 알림 — 규칙 0.5.7)
- `device = "mps" if torch.backends.mps.is_available() else "cpu"`(규칙 0.5.7) — 이 Mac은
  MPS 사용 가능, 전체 학습을 MPS로 돌렸다.
- 랜덤 시드: 42(`set_seed()`, `random`/`numpy`/`torch` 전부 고정). 증강 선택용 RNG도
  `np.random.default_rng(42)`에서 파생된다.

## 변경 파일 목록
```
M experiments/2nd/requirements.txt
M experiments/2nd/shaftlab/measure.py
A experiments/2nd/E7_edge1d/README.md
A experiments/2nd/E7_edge1d/build_cache.py
A experiments/2nd/E7_edge1d/augment.py
A experiments/2nd/E7_edge1d/model.py
A experiments/2nd/E7_edge1d/train_eval.py
A experiments/2nd/E7_edge1d/make_figures.py
A experiments/2nd/E7_edge1d/results/cv_metrics.csv
A experiments/2nd/E7_edge1d/results/setup_sigma.csv
A experiments/2nd/E7_edge1d/figures/clean_mae_by_combo.png
A experiments/2nd/E7_edge1d/figures/robustness_by_aug.png
A experiments/2nd/E7_edge1d/figures/setup_sigma.png
```
`cache/E7_edge1d/`(profiles.npz, oof_predictions.npz)와 `E7_edge1d/checkpoints/*.pt`는
`.gitignore`(`**/cache/`, `**/checkpoints/`, `*.pt`)에 의해 커밋되지 않는다 — 셋 다 스크립트
재실행으로 재생성된다(시드 고정, 단 oof_predictions.npz는 train_eval.py를 끝까지 돌려야
나온다). 5MB 초과 파일 없음.

## Codex adversarial-review 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] teacher 좌표가 모든 모델 입력·지름 재조립에 샌다 — `extract_profiles`가 프로파일 중심을 `round(teacher)`로 잡아, 라벨(중심→teacher 거리)이 반올림 오차(±0.5px)뿐인 순환 과제가 됨. 보고된 낮은 오차·σ로 cross50 대체/개선을 주장할 근거가 안 됨 | **맞음** — 코드로 확인(`tc=round(tops[k])`). 모델이 엣지를 찾는 게 아니라 teacher 자신의 반올림 오차만 보정하는 과제였다 | **반영함** | `extract_profiles`에 `coarse_seg` 옵션 추가(기본값 유지, 기존 동작 안 깨짐) — `coarse_section`의 1/4 해상도 teacher-무관 좌표를 중심으로 쓴다(새 라벨 범위: 상단 -5~0px, 하단 +5~+15px, 실측 확인). 과제가 훨씬 어려워져 모델 용량을 3층 5065 → 4층 34481 파라미터로 키웠다. 결과: heatmap 헤드는 여전히 완료 기준(≤0.05px)을 상단·하단 둘 다 통과(0.014~0.027px), 회귀 헤드는 둘 다 통과 못 함(0.088~0.177px) — 헤드 선택이 핵심 변수로 떠올랐다(위 "결과 요약"). |
| [medium] 상단 블러 σ=0.5가 E2의 "셋업최악" 기준으로는 게이트인데도 `TOP_ALLOWED`에 넣어 라벨 불변 가정을 약 1/5 학습 샘플에 주입 | **맞음** — E2 표를 셋업최악까지 다시 보니 상단은 테스트한 세 강도(0.5/1.0/2.0) 전부 게이트다. σ=0.25는 E2가 아예 테스트 안 한 값이었다(추가로 발견) | **반영함** | 상단 블러를 완전히 뺐다(`TOP_ALLOWED["blur_sigma"]=[]`). 하단은 전체중앙값·셋업최악 둘 다 허용인 σ∈{0.5,1.0}만 남겼다. `random_augment`도 상단일 때 "blur" 선택지가 애초에 빠지도록 고쳤다. |
| [medium] shift 증강이 81px로 자른 프로파일에 `CubicSpline(extrapolate=True)`를 써서, ±0.5px 이동 둘 다 창 반대쪽 끝에서 외삽이 필요 — 보고된 -0.5px 비대칭이 모델 약점이 아니라 보간 아티팩트일 수 있음 | **맞음** — 재현 확인(±0.5px 모두 창을 벗어나는 외삽 필요, 방향마다 반대쪽 끝) | **반영함** | `build_cache.py`가 89px(모델 입력 81px + 버퍼 4px씩)로 프로파일을 저장하고, `shift1d`가 이 버퍼 안에서만 보간하도록 고쳤다(필요 여유 0.5px, 버퍼 4px — 외삽 전혀 없음, `src range ... within [0,88]? True`로 수치 확인). 고친 뒤 heatmap 헤드의 shift 방향별 차이가 거의 사라졌다(top: -0.5px 0.0195 vs +0.25px 0.0151, 이전 0.061 vs 0.026의 3배 차이에서 크게 줄었다) — **Codex의 가설이 맞았다.** |
| [P2, 2026-10-16] "결과 요약 2" 표의 `max(px)` 열이 실제 최대 오차가 아니라 각 fold의 median \|오차\|(=`mae_px`) 중 최댓값임 — 예: `top/heatmap`의 0.035는 `cv_metrics.csv`의 진짜 `max_abs_err_px`(0.236)가 아니라 fold-median들의 최댓값. 꼬리 오차를 실제보다 작게 보고 | **맞음** — `cv_metrics.csv`에서 직접 계산 확인(`top/heatmap` 진짜 max=0.236, `bottom/회귀/flipped` 진짜 max=1.097, 베이스라인도 같은 문제: top 3.279→5.005, bottom 11.636→14.770) | **반영함** | 기존 열을 "fold 최악 median(px)"으로 이름을 바로잡고, 전체 fold의 진짜 `max_abs_err_px` 최댓값을 "실제 max(px)" 열로 새로 추가했다(위 "결과 요약 2" 표). 완료 기준(≤0.05px)은 median 기준이라 판정은 안 바뀌지만, 개별 열 단위 최악 오차는 이 열로 다시 봐야 한다. |
| [P2, 2026-10-16(2)] 89px 포맷 이전에 만든 81px `profiles.npz`가 남아 있으면 `A.crop_center`의 `[4:85]` 크롭이 조용히 77px만 반환한다 — 모델은 폭을 검증하지 않고 받아들이고, heatmap 헤드는 여전히 중심 인덱스 40을 가정하므로 에러 없이 학습·평가 지표가 틀어진 채로 나온다 | **맞음** — 코드로 확인(`crop_center`가 폭을 검증하지 않음). 실제로 합성한 81px 캐시로 재현: 수정 전 코드라면 조용히 77px을 반환했을 상황 | **반영함** | `train_eval.py`의 `load_cache()`에 폭 검증을 추가했다 — `top_profiles`/`bot_profiles`의 폭이 기대값(89px = `A.MODEL_WIDTH` + `A.SHIFT_BUFFER_PX`×2)과 다르면 `RuntimeError`로 즉시 멈추고 "캐시를 지우고 `build_cache.py`로 다시 만들라"는 메시지를 낸다. 현재 캐시(89px, 9347행)로 정상 통과 확인, 합성한 81px 캐시로 의도한 에러가 뜨는 것도 확인했다. |

동의하지 않는 지적 없음 — 다섯 지적 다 코드(또는 CSV 값)로 재현되는 실제 문제였다. 특히
첫 번째(teacher 좌표 누수)는 이 실험 전체의 설계를 바꾼 가장 중요한 수정이었다: 라벨
순환성을 없애자 과제 난이도가 급상승했고, 그래서 모델 용량을 키워야 했고, 그 결과
"헤드 구조(회귀 vs heatmap)가 핵심 변수"라는, 처음 버전에는 아예 보이지 않았던 결과가
나왔다. 네 번째(max(px) 오표기)는 완료 기준 판정 자체는 바꾸지 않지만, 보고서가 실제보다
낙관적인 꼬리 오차를 보여주고 있었다는 점에서 E10 평가 지표를 설계할 때 같은 실수(fold
median의 최댓값과 진짜 max를 섞는 것)를 반복하지 않아야 한다.

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- `coarse_seg` 기반 중심이 teacher와 "충분히" 독립적인지 — `coarse_section`과
  `edge_subpx`가 같은 원본 이미지의 같은 대략적 밝기 로직을 공유하는데, 이게 통계적
  독립성을 깨는 숨은 상관을 만들 가능성이 있는지 봐 달라(위 한계 마지막 항목).
- 회귀 헤드가 완료 기준을 통과 못 한 게 (a) 전역 평균 풀링 구조 자체의 한계인지,
  (b) 용량·epoch 부족인지(위 "결과 요약 1"의 수렴 추이가 250 epoch에도 안 끝났다),
  (c) 다른 이유인지 — heatmap과의 격차가 "구조가 본질적으로 유리하다"는 결론으로
  이어져도 되는지 봐 달라.
- Stage B(`stageB_both_heatmap_raw`)가 teacher보다 평균·최악 σ 둘 다 낮게 나온 걸
  (반복성 노이즈 바닥보다 작은 차이라는 단서를 붙여서도) "긍정적 신호"로 보고하는 게
  맞는지, 아니면 우연(13개 셋업, 재장착 없음)으로 치부하고 아예 언급을 자제해야 하는지.
- shift 강건성 수정(89px 버퍼)이 heatmap 헤드의 비대칭은 거의 없앴지만 회귀 헤드는
  여전히 방향별로 들쭉날쭉한데(top/회귀: -0.5px 0.194, +0.25px 0.102, +0.5px 0.179),
  이게 회귀 헤드 자체의 노이즈(애초에 정확도가 낮아 모든 지표가 불안정)인지 남은 구조적
  문제인지 봐 달라.
- 모델 용량(4층, 34.5천 파라미터)·epoch(200)을 top/회귀 1개 조합의 수렴 추이만 보고
  정해서 나머지 5개 조합에 그대로 적용했는데, 조합마다(특히 하단) 따로 튜닝했어야
  하는지 — 하단 회귀가 상단 회귀보다도 더 나쁜 이유(0.144~0.177 vs 0.088)가 단순히
  더 넓은 라벨 범위(10px vs 5px) 때문인지, 하단 특유의 다단 구조 때문인지도 궁금하다.
