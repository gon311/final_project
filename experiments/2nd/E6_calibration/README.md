# E6. 캘리브레이션 방식 비교 (평균 찍기 기준선 포함)

## 한 줄 결론
**평균 찍기를 "확실히" 이기는 조합은 없다 — 이 보고서의 수치는 전부 탐색 결과이지 방법
선택 근거가 아니다(동일 장착 10장 반복만 봤다, 재장착 반복 없음, 규칙 8).** 가장 비슷하게
근접한 건 **grad_peak + 방식2(전체 선형회귀)**로, 사진 선택 3종·평가 분할 2종(6개 조합)
중 5개에서 평균 찍기보다 근소하게 낮은 MAE를 냈고 나머지 1개도 거의 동률이었다
(−1.2\~+0.06µm) — 정상품 오판정은 6개 조합 전부 0건이었다. **하지만 이 차이(최대 \~1µm)는
같은 장착 10장 반복 자체의 흔들림(cross50 3.4µm·grad_peak 5.0µm·moment 4.6µm,
`repeatability_check.py`)보다도 작다** — 재장착 없이는 "방법이 좋아서"인지 "촬영 노이즈
안에서 우연히 낮게 나온 것"인지 가릴 수 없다는 뜻이다. 나머지 조합(특히 물리스케일 고정,
3점 선형을 우리 정의로 다시 맞춘 경우)은 평균 찍기보다 뚜렷이 나쁘고 정상품을 NG로 잘못
판정하는 경우가 흔하다. **재현 체크포인트는 통과했다** — 팀원A V3 엔진 px로 NO4\~6 3점
선형을 맞추면 기울기 2.586µm/px(팀원B 재현값 약 2.57), NG1·NG2 20장 전부 OK로 정확히
일치한다.

## 한 일
1. `scan_values.py`: 베어링부 cross50·grad_peak는 **다시 재지 않고** `eda_measure_raw.csv`
   (E1이 재현한 원본, step=2)의 `B_d_line`/`B_d_peak_median`을 그대로 쓴다. 핀 cross50·
   grad_peak는 E5가 고른 `measure_pin_relative()`로 새로 잰다(기존 `pin_gauge_raw.csv`는
   "auto" 위치 방식 기준이라 쓸 수 없다). moment_centroid(E4의 두 번째 teacher 후보,
   "E4 최선")는 베어링부·핀 둘 다 새로 잰다(기존 CSV에 없는 정의라서). 산출물:
   `results/values_by_image.csv`(130행).
2. `calibration.py`(공용 엔진) + `run_calibration.py`: 방식 0·2·3·4 x 정의 3종 x 사진 선택
   3종 x 분할 2종을 전부 돌린다. 방식 1(3점 선형)은 항상 묶음4·5·6만 학습해 따로 다룬다.
   산출물: `results/calibration_loso.csv`(방식×정의×사진선택×분할 요약, 90행),
   `results/calibration_by_specimen.csv`(단위별 상세, 참고).
3. `teamA_checkpoint.py`: 지시서의 "재현 체크포인트"(팀원A 엔진 경로로 NO4\~6 3점 선형 →
   기울기 약 2.57µm/px, NG1·NG2 20장 전부 OK)를 검증한다. **다시 계산하지 않고** E4가 이미
   돌려 커밋해 둔 `E4_edge_benchmark/results/reference_compare.csv`의
   `teamA_v3_measured_px`를 그대로 쓴다.
4. `focus_group_fit.py`: 방식5((참고) 초점 구간별 적합) — F1/F2/F3 안에서만 선형 적합,
   구간당 묶음이 2\~5개뿐이라 in-sample 잔차만 본다. **다른 스크립트와 달리 "시편" 대신
   "묶음" 단위로 묶는다**(Codex `/codex:review` 반영) — 초점은 묶음(촬영)의 속성이라
   NO1(묶음1·13)이 F1·F3에 각각 따로 들어가야 맞다.
5. `repeatability_check.py`(신규, Codex adversarial-review 반영): 재장착 반복이 없으니(규칙
   8) 최소한 **같은 장착 10장 반복 프레임의 흔들림**을 숫자로 내 둔다(재장착 변동의 대리값은
   아니다, 아마 그보다 작을 "하한"이다) — 방식 간 MAE 차이를 이 숫자와 나란히 읽으라는
   뜻이다. 산출물: `results/within_mount_repeatability.csv`.
6. `make_figures.py`: 그림 2장(아래 "그림") — 둘 다 "동일 장착 반복일 뿐, 방법 선택 근거
   아님" 경고 문구를 그림 자체에 넣었고(Codex adversarial-review 반영), 막대그래프에는
   5번의 반복성 수치를 회색 음영대로 같이 그렸다.

## 방법론 메모
- **평가는 항상 셋업(시편) 단위**(규칙 3) — 이미지 10장을 그대로 회귀에 넣지 않고, 셋업
  대표 px(사진 선택에 따라 10장 전부/마지막 5장의 중앙값)를 먼저 낸 뒤 그 값 하나로
  학습한다.
- **NO1 중복(0.3절)**: 같은 시편 NO1이 묶음1(P1)과 묶음13(P2 재촬영)에 있다. 시편 단위
  LOO에서는 둘을 "NO1" 한 단위로 합쳐서(이미지들을 합쳐 대표 px를 내고) 함께 뺀다.
- **방식4(핀 비율)에서 묶음1은 항상 학습 제외**다(지시서, 핀이 P1 구간에서 한 번 이동했기
  때문). NO1 자체는 "학습에 한 번도 못 들어가지만 나머지 9개로 맞춘 모델로 평가는 받는"
  특별 취급이다(핀비율 정의상 당연한 처리이지 NO1을 차별하는 게 아니다).
- **그림(`cross50_px_vs_mm.png`)에서 NG1·NG2는 방식2·3 적합에 넣지 않았다** — 처음 버전은
  실수로 13개 셋업(NG 포함) 전체로 "참고용" 추세선을 그렸는데, 그러면 규칙4(NG 미사용
  학습)를 그림에서만 어기는 꼴이라 정상 10개로만 다시 맞췄다(스스로 발견해 고침).

## 결과 요약

### 1단계 — 재현 체크포인트 (`teamA_checkpoint.py` 콘솔 출력)
| 항목 | 우리 결과 | 팀원B 재현값 |
|---|---|---|
| 기울기 | **2.586 µm/px** | 약 2.57 µm/px |
| NG1·NG2 20장 OK | **20/20** | 20/20 |
**정확히 일치한다.** 이게 통과하지 않으면 아래 결과 전체를 믿기 어려웠을 텐데, 통과했으니
나머지 비교(우리 자체 엣지 정의)도 같은 파이프라인·같은 공차 규칙으로 믿을 수 있다.

### 2단계 — 핵심 수치 표 (`results/calibration_loso.csv`, 사진선택=all10, 규칙 10: 첫 줄=평균 찍기)
**분할 A(시편 LOO):**
| 정의 | 방식 | MAE(µm) | 평균찍기 대비 | 정상 오판정 | NG1 | NG2 |
|---|---|---|---|---|---|---|
| (모든 정의) | **0_평균찍기** | **5.78** | 0 | 0 | OK | OK |
| cross50 | 2_전체선형회귀 | 7.22 | +1.44 | 0 | OK | OK |
| cross50 | 3_물리스케일고정 | 26.88 | +21.10 | 7 | NG | NG |
| cross50 | 1_3점선형(NO4\~6) | 14.18 | +8.40 | 5 | NG | NG |
| cross50 | **4_핀비율정규화선형** | **3.95** | **−1.82** | 1 | OK | OK |
| grad_peak | **2_전체선형회귀** | **5.68** | **−0.10** | 0 | OK | OK |
| grad_peak | 1_3점선형(NO4\~6) | 5.07 | −0.70 | 0 | OK | OK |
| moment | **4_핀비율정규화선형** | **4.62** | **−1.15** | 1 | OK | OK |

**분할 B(팀원A 분할, NO4\~6 학습 → NO7\~10 예측):**
| 정의 | 방식 | MAE(µm) | 평균찍기 대비 | 정상 오판정 | NG1 | NG2 |
|---|---|---|---|---|---|---|
| (모든 정의) | **0_평균찍기** | **3.50** | 0 | 0 | OK | OK |
| grad_peak | **2_전체선형회귀** | **2.47** | **−1.03** | 0 | OK | OK |
| grad_peak | 1_3점선형(NO4\~6) | 2.47 | −1.03 | 0 | OK | OK |
| grad_peak | 4_핀비율정규화선형 | 2.47 | −1.03 | 0 | OK | OK |
| cross50 | 4_핀비율정규화선형 | 7.73 | +4.23 | 3 | OK | OK |
| cross50 | 1_3점선형(NO4\~6) | 17.28 | +13.78 | 3 | NG | NG |
| cross50 | 2_전체선형회귀 | 17.28 | +13.78 | 3 | NG | NG |

**방식1·방식2가 분할B에서 숫자가 똑같은 건 우연이 아니라 구조적으로 같은 계산이기
때문이다** — 분할B의 학습 집합이 애초에 "묶음4·5·6"(방식1의 고정 정의 그 자체)이라서,
"전체 선형회귀"(방식2)도 같은 3점으로 회귀하면 수학적으로 동일한 직선이 나온다(코드로
`slope`·`intercept`가 소수점까지 일치함을 확인했다). 분할A에서는 방식2가 10개(LOO로 9개)
전체로 맞춰서 방식1과 달라진다 — 그래서 분할A 표에만 방식1·2가 따로 다른 값으로 나온다.

- **cross50·moment + 핀비율정규화선형은 분할 A에서는 기준선을 이기지만 분할 B에서는 진다**
  (cross50: −1.82µm → +4.23µm). **이건 "방향이 뒤집히는" 전형적인 과적합 신호다** — E4에서
  esf2_outer 상관이 n을 늘리자 무너진 것과 같은 함정이다. 그래서 "이긴다"고 결론 내리지
  않는다.
- **grad_peak + 전체선형회귀는 6개(사진선택3×분할2) 조합 중 5개에서 기준선보다 낮았고,
  나머지 1개(사진선택=no7excl, 분할A)도 +0.06µm로 사실상 동률이었다**(−1.16\~+0.06µm 범위,
  `results/calibration_loso.csv`로 전체 확인 — 가장 나쁜 경우도 기준선과 차이가 거의
  없었다는 뜻이다). 정상 오판정도 6개 조합 전부 0건, NG1·NG2도 항상 올바르게 OK. 그래도
  마진이 작고(공차 반폭 0.27px≈10µm보다 작다), 13개 셋업·재장착 반복 없음(규칙 8)이라
  "확실한 승리"라고 못 박지 않는다. **구체적으로 얼마나 작은지 숫자로 대 보면** —
  `repeatability_check.py`로 **같은 장착 10장 반복 자체의 흔들림**(재장착 변동이 아니라
  그보다 작을 "하한"일 뿐이다)을 재 보니 cross50 3.4µm·grad_peak 5.0µm·moment 4.6µm
  (`results/within_mount_repeatability.csv`). **방식 간 MAE 차이(최대 \~1µm)가 이 흔들림
  자체보다 한참 작다** — `figures/mae_by_method.png`의 회색 음영대가 그 비교를 바로
  보여준다. 즉 "grad_peak+전체선형회귀가 평균 찍기를 이긴다"는 관찰은 같은 장착 안에서도
  나타날 수 있는 변동 폭 안에 있어, 이 데이터만으로는 방법 차이인지 촬영 노이즈인지 가릴
  수 없다(Codex adversarial-review 반영).
- **물리스케일 고정(방식3)은 모든 정의·분할에서 명백히 최악**이다 — MAE 11\~64µm, 정상
  시편을 최대 8/10개까지 NG로 잘못 판정한다. 0.2절의 "스케일 0.0367mm/px은 알려진 값이지만
  고정 오프셋 하나로는 안 맞는다"는 사실과 일치.
- **우리 정의로 다시 맞춘 3점 선형(방식1)의 기울기는 cross50 +10.1, grad_peak +2.6,
  moment +12.9 µm/px — 전부 양수다.** 팀원B가 자기 정의(M0 등)로 다시 맞췄을 때 음수
  (−25.5, −8.4µm/px)를 얻었다는 0.2절 기록과 **부호가 다르다.** 흥미롭게도 **grad_peak의
  +2.6µm/px은 팀원A V3 엔진 재현값(+2.586µm/px)과 거의 같다** — E4에서 이미 본 "grad_peak와
  V3가 NO4에서 같이 튄다"는 사실과 같은 방향(둘 다 "가장 센 그래디언트 쌍"을 쫓는 정의라서
  비슷하게 행동하는 것으로 보인다, 가설).

### 3단계 — 사진 선택 변형 (last5 vs all10 vs no7excl)
`results/calibration_loso.csv`에서 같은 정의·방식·분할로 사진 선택만 바꿔 보면 **차이가
1µm 안쪽으로 작다**(예: grad_peak+전체선형회귀, 분할A: all10 5.68 / last5 5.28 / no7excl
6.34). **팀원B가 보고한 "마지막 5장이 11.1µm로 뚜렷이 좋아진다"는 효과가 우리 데이터·정의
에서는 재현되지 않는다.** 엣지 정의 세부 구현(샘플링 창 등)이 다른 게 원인일 수 있다 —
단정하지 않는다.

### 4단계 — (참고) 초점 구간별 적합 (`results/focus_group_fit.csv`)
**초점은 시편이 아니라 묶음(촬영)의 속성이라, 이 분석만 "묶음" 단위로 묶는다** — NO1은
묶음1(F1)과 묶음13(F3) 두 번 찍혔으니, F1에 한 번·F3에 한 번 각각 독립 데이터점으로
들어간다(다른 분석들처럼 "시편" 단위로 둘을 합치면 "그 시편의 초점이 뭐냐"는 질문 자체가
성립하지 않는다 — Codex `/codex:review` 반영, 처음엔 묶음13을 통째로 걸러서 F3에서 NO1 몫이
빠져 있었다). F1(묶음1\~4, n=4)·F3(묶음7\~10·13, **n=5**) 안에서 in-sample 잔차는 2.1\~7.8µm로
구간에 따라 크게 다르다. F2는 묶음이 2개뿐이라 잔차가 항상 0(직선이 두 점을 지나니 당연하다,
무의미). **이 수치들을 다른 방식의 LOO MAE와 비교하지 않는다** — in-sample이라
과소평가돼 있다(지시서 "(참고)" 표기와 같은 이유).

## 팀원B 결과와 대조
| 항목 | 일치 여부 |
|---|---|
| 팀원A 경로 재현: 기울기 2.57µm/px, NG1·NG2 20/20 OK | **일치**(우리 2.586µm/px, 20/20 OK) |
| 우리 정의로 3점 다시 맞추면 기울기 음수 | **불일치**(우리는 cross50/grad_peak/moment 전부 양수) |
| 어떤 방법도 평균 찍기 못 이김(M0 LOO 15.5µm > 기준 5.8µm) | **방향 일치**: 우리도 대부분 못 이긴다, 다만 grad_peak+전체선형회귀는 근소하게 이김(과신 안 함) |
| 마지막 5장이 11.1µm로 뚜렷이 좋아짐 | **불일치**: 우리 데이터에서는 1µm 안쪽 차이뿐 |
| 물리스케일 고정류가 특히 나쁨 | **일치**: 우리도 방식3이 전부 최악 |

## Codex adversarial-review 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] 분할B의 방식1(3점선형)이 NO7\~10이 아니라 분할A의 평가 대상(NO1·B2·B3·B7\~10, 7개 정상 시편)을 그대로 재사용해 분할B 결과가 오염됨 | **맞음** — `run_calibration.py:153~155`가 `rows1`(분할A용)을 분할B 요약에도 그대로 넣고 있었다. 실제 산출 CSV의 분할B 방식1 행 `n_normal=7`이 증거였다 | **반영함** | 분할B 전용 `rows1_b = evaluate_fold(predict_fn1, setup_b.loc[eval_units_b])`를 새로 계산해 NO7\~10(또는 no7excl일 땐 있는 것만)만 평가하도록 고쳤다. 고치면서 `predict_fn`이라는 변수명을 방식0·2·3 루프가 마지막(방식3)으로 덮어쓰고 있다는 걸 직접 발견해 `predict_fn1`로 분리했다 — 그대로 뒀으면 "고친 코드"가 실제로는 물리스케일고정 적합을 재사용하는 새 버그가 됐을 것이다. 수정 후 분할B 방식1은 cross50 MAE 14.18→17.28µm, grad_peak 5.07→2.47µm, moment 20.04→24.71µm로 분할A와 다른(정확한) 값이 나온다. 또한 분할B에서 방식1과 방식2가 수치가 똑같이 나오는데, 이건 버그가 아니라 분할B의 학습 집합 자체가 "묶음4·5·6"이라 두 방식이 수학적으로 같은 계산이 되기 때문임을 확인해 README에 설명을 추가했다. |
| [medium] 규칙8(재장착 반복 없음) 미충족 상태에서 단일 장착 반복만으로 방법 우열을 제시 — CSV·그림이 일반화 성능으로 오해될 수 있음 | **맞음** | **반영함** | (1) `repeatability_check.py`를 신설해 "같은 장착 10장 반복" 자체의 흔들림(cross50 3.4µm·grad_peak 5.0µm·moment 4.6µm)을 수치화했다 — 방식 간 최대 차이(\~1µm)가 이보다 작다는 걸 직접 비교할 수 있게 했다. (2) `calibration_loso.csv`의 모든 행에 `검증_범위="동일장착내_재장착반복없음(규칙8)"` 열을 추가했다(README뿐 아니라 CSV 자체에 남김). (3) 두 그림 모두에 "동일 장착 반복 결과, 방법 선택 근거 아님" 경고 문구를 그림에 직접 넣었고, `mae_by_method.png`에는 반복성 수치를 회색 음영대로 겹쳐 그려 시각적으로도 비교되게 했다. (4) "한 줄 결론"·2단계 bullet을 다시 써서 "탐색 결과"라는 프레이밍을 명시했다. 독립 재장착 자체는 이번 촬영 데이터로는 추가할 수 없어 다음 촬영 과제로 남겼다(한계에 기록). |

동의하지 않는 지적 없음 — 둘 다 코드·수치로 재현되는 실제 문제였다.

## Codex `/codex:review` 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] `focus_group_fit.py`가 NO1(묶음1·13)을 "시편" 단위로 합쳐서, F3 그룹에 들어가야 할 묶음13(F3 재촬영)이 실제로는 F1로 잘못 분류되거나 두 초점이 섞임 | **결론은 맞지만 지적한 메커니즘은 코드와 달랐다** — Codex는 `.mode()`가 F1을 골라 median이 F1·F3을 섞는다고 설명했지만, 실제로는 그보다 앞선 `df[df["묶음"]<=10]` 필터가 묶음13 자체를 지워서 섞일 데이터가 애초에 없었다(섞이는 게 아니라 빠지는 것). 어느 쪽이든 "F3 그룹에 NO1의 F3 촬영분이 없다"는 결론은 코드로 확인되는 사실이었다 | **반영함** | "시편" 대신 "묶음" 단위로 다시 짰다 — 초점은 묶음(촬영)의 속성이지 시편의 속성이 아니므로, 묶음1(F1)·묶음13(F3)을 각각 독립 데이터점으로 둔다(NG만 걸러내고 묶음13은 유지). F3 묶음 수 4→5로 늘었고, in-sample MAE가 cross50 1.54→2.13µm, grad_peak 1.57→2.10µm, moment 1.27→2.24µm로 바뀌었다(F1은 애초에 안 섞였어서 불변). |
| [P2] `run_calibration.py`의 방식4(핀비율정규화선형) 분할B 블록이 `summary_rows`에만 쓰고 `detail_rows`에 안 넣어 `calibration_by_specimen.csv`에서 방식4/분할B 행이 통째로 빠짐 | **맞음** — 다른 모든 방식·분할 블록은 둘 다 쓰는데 이 블록만 `detail_rows` 추가 줄이 없었다 | **반영함** | 누락된 `detail_rows += [...]`를 추가했다. 재생성 후 `calibration_by_specimen.csv`가 702→753행으로 늘었고(+51행), 정의 3종×사진선택 3종×(B7\~12 또는 no7excl이면 B8\~12) 조합이 전부 들어있음을 확인했다. |

동의하지 않는 지적 없음 — 둘 다 코드·수치로 재현되는 실제 문제였다. 첫 번째 지적은 결과(결론)에는 동의하지만 Codex가 설명한 원인 메커니즘은 실제 코드 흐름과 달랐다는 점만 위 표에 정정해 남겼다.

## 그림
- `figures/mae_by_method.png`: 방식별 LOO MAE 막대그래프(사진선택=all10, 분할=A), 평균
  찍기 기준선을 점선으로, **같은 장착 10장 반복성(3.4\~5.0µm)을 회색 음영대**로 같이 표시 —
  막대 차이가 이 밴드보다 작으면 방법 우열이 아니라 촬영 노이즈 안일 수 있다는 걸 바로
  보여준다(Codex adversarial-review 반영).
- `figures/cross50_px_vs_mm.png`: cross50 셋업 중앙값 px vs 실측mm 산점도 + 방식2·3 적합선
  (정상 10개로만 적합, NG1·NG2는 평가 참고용 X 표시로만 — 규칙4).
- 두 그림 모두 아래쪽에 "동일 장착 10장 반복 결과 — 재장착 반복 검증 없음, 방법 선택
  근거 아님" 경고 문구를 직접 넣었다(Codex adversarial-review 반영).

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음(규칙 8)** — grad_peak+전체선형회귀가 6개 조합 중 5개에서
  기준선을 이기고 나머지 1개도 거의 동률인 것도 "이 촬영 1회의 반복성" 안에서일 뿐이다.
  재장착을 다시 하면 바로 뒤집힐 수 있다.
- **moment_centroid는 이번에 처음 쟀다**(E4에서 추가된 정의) — cross50/grad_peak처럼
  원본 CSV와 대조해 검증한 적이 없다. E4의 셋업 내 σ 비교(cross50과 비슷하게 안정적)만
  근거로 믿고 썼다.
- **방식4(핀비율)의 split 간 부호 역전(cross50·moment)은 과적합 신호로 해석했지만, 다른
  원인(예: 핀·샤프트가 서로 다른 평면에 있어 카메라 이동 영향을 다르게 받는다, E5에서 이미
  나온 가설)일 가능성도 있다** — 이 실험만으로는 둘을 못 가른다.
- **사진 선택 변형(last5)이 팀원B처럼 뚜렷한 개선을 안 보인 이유를 못 밝혔다** — 엣지
  정의 차이인지, 팀원B 쪽 특정 방법(M0)에서만 나타나는 효과인지 확인하려면 같은 정의로
  나란히 돌려야 한다(다음 촬영 또는 추가 분석 과제).
- **NG1·NG2는 평가에만 썼다**(규칙 4) — 어느 방식·정의도 학습에 NG를 넣지 않았다(그림도
  마찬가지로 고침, 위 "방법론 메모").
- 확정은 다음 촬영(핀게이지 세트 GT, 재장착 반복)으로 한다(지시서 요구사항).

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate   # 없으면: python3.11 -m venv .venv && pip install -r requirements.txt
cd E6_calibration
python3 scan_values.py          # results/values_by_image.csv
python3 run_calibration.py      # results/calibration_loso.csv, calibration_by_specimen.csv
python3 teamA_checkpoint.py     # 콘솔 출력만(재현 체크포인트)
python3 focus_group_fit.py      # results/focus_group_fit.csv ((참고) 방식5)
python3 repeatability_check.py  # results/within_mount_repeatability.csv
python3 make_figures.py         # figures/*.png (within_mount_repeatability.csv 먼저 있어야 함)

# E1 영향 재확인(E6은 shaftlab을 수정하지 않았다 — 기존 함수만 호출)
cd ../E1_module && python3 reproduce.py
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, matplotlib 3.11.2,
  scipy 1.17.1(E4에서 추가됨, E6은 쓰지 않음)
- torch: 사용하지 않음
- 랜덤 시드: 이 실험은 순열검정·셔플이 없어 고정할 난수가 없다(LOO는 결정론적).

## 변경 파일 목록
```
A experiments/2nd/E6_calibration/README.md
A experiments/2nd/E6_calibration/scan_values.py
A experiments/2nd/E6_calibration/calibration.py
A experiments/2nd/E6_calibration/run_calibration.py
A experiments/2nd/E6_calibration/teamA_checkpoint.py
A experiments/2nd/E6_calibration/focus_group_fit.py
A experiments/2nd/E6_calibration/repeatability_check.py   (Codex adversarial-review 반영, 신규)
A experiments/2nd/E6_calibration/make_figures.py
A experiments/2nd/E6_calibration/results/values_by_image.csv
A experiments/2nd/E6_calibration/results/calibration_loso.csv   (검증_범위 열 추가)
A experiments/2nd/E6_calibration/results/calibration_by_specimen.csv
A experiments/2nd/E6_calibration/results/focus_group_fit.csv
A experiments/2nd/E6_calibration/results/within_mount_repeatability.csv   (Codex adversarial-review 반영, 신규)
A experiments/2nd/E6_calibration/figures/mae_by_method.png
A experiments/2nd/E6_calibration/figures/cross50_px_vs_mm.png
```
`shaftlab/`은 전혀 건드리지 않았다(기존 함수만 호출). 5MB 초과 파일 없음(가장 큰 산출물은
`calibration_by_specimen.csv` 84KB 안팎, 그림 두 장은 각 80KB 이하).

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- 방식4(핀비율정규화선형)의 cross50·moment에서 분할 A는 기준선을 이기고 분할 B는 지는
  부호 역전을 "과적합"으로 해석했는데, 이게 맞는 해석인지, 아니면 둘 다 보고하고 결론을
  유보하는 게 더 정직한지 봐 달라.
- grad_peak+전체선형회귀가 6개 조합 중 5개에서 기준선을 이기고 나머지 1개도 거의 동률인
  걸 "확실한 승리로 보지 않는다"고 했는데, 이 판단 기준(마진이 공차 반폭보다 작다, 재장착
  반복 없음)이 충분히 명확하고 일관되게 적용됐는지 봐 달라.
- `run_calibration.py`의 방식4 LOO(`exclude_from_train={"NO1"}`)가 NO1을 "학습 9개로 맞춘
  모델로 평가만" 하는 방식이 지시서의 "묶음1은 제외하고 맞춘다"는 요구를 올바르게 구현한
  건지, 혹시 NO1을 아예 평가에서도 빼야 하는 건지 봐 달라.
- moment_centroid px를 새로 재면서 E4와 다른 step(4px, E4는 bearing 10px)을 썼는데, 이
  차이가 cross50/grad_peak와의 비교 공정성에 영향을 주는지 봐 달라.
- `repeatability_check.py`가 "같은 장착 10장 반복 σ"를 방법 차이의 비교 잣대로 쓰는 게
  통계적으로 타당한 접근인지(단순 비교일 뿐 정식 유의성 검정은 아니다), 아니면 순열검정
  같은 더 formal한 검정이 필요한지 봐 달라.
