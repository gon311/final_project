# E5. Ø10 핀게이지 심화 분석 (핀 측정 위치 결정)

## 목적
(1) 이후 실험이 쓸 핀 측정 위치를 정한다. (2) 핀을 셋업 모니터(초점·배율·카메라 이동 감지)로
쓸 수 있는지 판정한다. (3) 핀과 샤프트의 스케일 불일치 원인 후보를 좁힌다.

## 한 일
0. `experiments/2nd/requirements.txt`에 `matplotlib`을 추가(그림 산출물용).
1. `shaftlab/measure.py`에 `measure_pin_profile()`을 추가해 핀 구간 전체(모따기·V블록
   그림자는 `coarse_section`의 높이 필터로 이미 제외됨)의 열별 cross50·grad_peak 엣지를
   둘 다 뽑을 수 있게 했다(`measure_section`은 수정하지 않고 별도 함수로 분리).
   또한 `measure_section`의 내부 루프를 `_measure_window(full, seg, xa, xb, step)`로
   뽑아내고 `measure_section`은 margin 적용 후 이 함수를 호출하도록만 리팩토링했다
   (산술은 동일, E1 재현 결과로 확인 — 아래 "E1 영향" 참고). `measure_pin()`의
   `position_method="pin_relative"` 창은 검출된 핀 범위(`xR`)로 클램프하고, 잘렸으면
   `window_clipped=True`를 반환한다(Codex 2차 adversarial-review 반영, 아래 "Codex 2차
   반영" 참고).
2. `shaftlab/plotting.py`에 `setup_korean_font()`를 추가했다(한글 라벨이 □로 깨지는 문제
   해결, Codex 2차 리뷰 추가 요청). macOS `AppleGothic` -> `Apple SD Gothic Neo` ->
   `NanumGothic` -> `Malgun Gothic` 순으로 `matplotlib.font_manager`에 실제 설치된 폰트만
   골라 `rcParams`에 설정하고, 하나도 없으면 경고만 남기고 `None`을 반환한다(호출부가 영어
   라벨로 대체).
3. `scan_pins.py`: 분류표에 시편 정보가 있는 130장(0.3절, 메모 사진 제외)을 한 번만
   디코딩해 `results/pin_profile.csv`(1단계 산출물, 10650행=130장×열)와
   `cache/pin_bearing_scalars.csv`(커밋 안 함, 이후 단계가 재사용하는 스칼라 요약)를 만든다.
4. `step1_profile_check.py` ~ `step6_setup_monitor.py`: 아래 "결과 요약" 참고.

## 결과 요약

### 1단계 — 핀 폭 프로파일 재현 (`results/pin_slope_by_bundle.csv`, `figures/pin_width_profile.png`)
- 위 경계 기울기(cross50, 13묶음): **-2.03 ~ -1.24 px/100px** (팀원B: -1.72~-1.30). 방향·크기
  대체로 일치하나 우리 쪽이 범위가 조금 더 넓다(묶음11은 pin_len이 짧게 잡힌 프레임이 많아
  기울기 추정이 덜 안정적).
- 아래 경계 기울기: **-0.25 ~ +0.08 px/100px** (팀원B: -0.14~+0.24). 거의 평평하다는 결론은
  같지만, 우리 쪽이 약간 더 음의 방향으로 치우쳐 있다.
- s-구간별 평균 폭(자기 중앙값 뺀 값): s 140~159에서 +1.71px(팀원B +1.27), s 160~179에서
  +1.83px(팀원B +1.96) — **"핀 길이 끝(V블록 쪽)으로 갈수록 폭이 커진다"는 팀원B 결과가
  방향·크기 모두 재현된다.**

### 2단계 — 핀 측정 위치 결정 (`results/pin_position_compare.csv`, `figures/pin_position_methods.png`)
추정 방식(강건 직선적합, `measure_section`과 동일 로직)은 세 방식 모두 같게 두고 **측정
창(열 구간)만** 바꿔 비교했다. **세 방식 모두 `shaftlab.measure_pin()`을 130장에 직접
호출해서 얻는다**(캐시된 `pin_profile.csv`를 다시 집계하지 않음 — Codex 2차 review 반영,
런타임 API와 평가 산출물이 다른 계산이 되는 걸 막는다).

| 방식 | 묶음 안 sd(중앙값) | F1 묶음간 범위 | F2 묶음간 범위 | F3 묶음간 범위 | 묶음1·2 vs 3+ 차이 |
|---|---|---|---|---|---|
| (a) 현 EDA 자동 구간 | 0.076px | 0.661px | 0.027px | 0.235px | -0.752px |
| (b) 고정 열 9655~9700 | 0.090px | **1.158px** | 0.031px | 0.110px | -0.303px |
| (c) 핀 상대 s=60~105 | **0.077px** | 0.687px | 0.064px | 0.173px | -0.756px |

- **창 클램프 영향:** `window_clipped=True`(창이 `xR`에 잘림)인 프레임은 묶음2의 2장뿐이다
  (`pin_len=96px`로 요청한 s=60~105 구간을 다 못 채움). 클램프 전/후로 그 2장의 값은
  269.61/269.71px로 **거의 바뀌지 않았다**(클램프 전 버그는 `_measure_window`가 범위를
  넘는 열에서 `seg`의 마지막 행을 재사용해 작동은 했으나, 핀 바깥 신호가 섞일 수 있는
  위험이 있었다 — 이번 데이터에서는 그 위험이 수치로 드러나지 않았을 뿐이다). **이 2장은
  하류(3·5·6단계)의 묶음 평균·흐름 계산에서 뺐다**(Codex 3차 review 반영, 아래 "window_clipped
  전파" 참고) — 뺀 뒤 묶음2 평균은 269.635px(빼기 전 269.641px, 차이 0.006px로 미미).

- **(c) 핀 상대 구간을 `shaftlab.measure_pin`의 기본값으로 선정했다.** 묶음 안 sd가 가장
  작고, F1 묶음간 범위도 (a)와 거의 같다(둘 다 묶음 1~4 사이 카메라가 실제로 여러 번
  움직였다는 사실 자체를 반영하는 값이라 완전히 없어지지는 않는다). (b) 고정 열은 F1 묶음간
  범위가 가장 나쁘다 — 카메라가 밀리면(묶음1·2는 xL이 9560·9576으로 묶음3+의 9594~9616보다
  훨씬 왼쪽) 고정 열이 핀의 다른 부위(묶음에 따라 s=47~140 범위 어디든)를 재게 되기 때문이다.
  이는 팀원B 결론(그들의 B=핀 상대 방식이 A=고정 열보다 낫다, `1011_pincol` 3절)과 같은 방향이다.
- **(c)를 고른 또 다른 이유(0.5절 규칙 14):** 고정 열은 다음 촬영 구도가 바뀌면 핀을 벗어날
  수도 있다. 핀 상대 구간은 핀이 화면 어디에 있어도 같은 상대 위치를 잰다.
- 묶음1·2 vs 3+ 차이는 세 방식 모두 -0.35~-0.76px로 남는다 — 측정 위치를 고치는 것만으로는
  이 점프(실제 카메라 이동, 4단계 참고)를 없앨 수 없다.

### 3단계 — 핀 이동 시점 (`results/pin_state.csv`)
- 묶음1 안(10장)에서 xL은 9560~9564로 4px 안에서 안정 — **묶음1 "안에서" 핀이 움직인
  흔적은 보이지 않는다.**
- 묶음1→묶음2 raw xL 변화 +16px. 팀원B 위상상관(오른쪽V블록, 핀과 무관한 고정물) 기준
  카메라 이동 성분은 +12.4px(인용, `1011_pincol/_out/shift.txt`). **카메라 이동을 빼도
  잔여 핀 이동 +3.6px가 남는다.**
- **결론: 분류표의 "묶음 1 안에서 이동"(P1)보다 팀원B의 "묶음 1과 2 사이"(현장 LOG 70) 해석이
  우리 데이터와 더 맞는다.** `pin_state` 열은 묶음1=`before_move`(10행), 묶음2~13=`after_move`
  (120행)로 기록했다(분류표 해석과 다르게 기록, 0.3절 지시대로).

### 4단계 — 카메라 이동 대조 (`results/camera_shift_table.csv`)
팀원B `1011_camstate/_out/state.csv`(받침대 경계, 핀과 무관한 고정물)를 읽어 **묶음 경계
(이전 묶음 마지막 장 -> 이 묶음 첫 장) 기준** 변화량을 만들고, 우리가 잰 핀 xL·샤프트
위치(x_from, top_y, bot_y)의 같은 기준 변화를 나란히 뒀다. **이전 묶음 "첫 장"과 비교하는
방식(첫-첫 비교)은 쓰지 않았다** — 이전 묶음 안의 흐름·드리프트가 전환 이동으로 잘못
섞일 수 있어서다(Codex 2차 review 반영). 첫-첫 비교값은 `*_first_to_first` 열에 진단용으로
따로 남겼다(예: 묶음7은 올바른 기준으로 `delta_ours_shaft_x_from=-44px`인데 첫-첫 비교로는
`-3px`로 전혀 다르게 보인다 — 묶음6 안의 드리프트가 섞인 예시다).
- 대부분의 묶음 전환에서 받침대 경계가 몇 px~40px 가까이 움직였다(팀원B 결론과 일치).
- **묶음9→10(NO9→NO10), 묶음10→11(NO10→NG1)에서 우리 샤프트 y가 각각 -17px, +19px
  뛰었는데, 팀원B 고정물의 세로(yB) 변화는 각각 +1.7px, +0.5px로 작다.** 카메라 이동으로
  설명되지 않는 이 잔여 이동을 `residual_note` 열에 **"시편 놓임(앉은 높이) 후보"**로
  표시했다(0.2b의 "앉은 높이" 오차 후보와 같은 방향). 기준을 고쳐도 이 두 묶음의 판정은
  바뀌지 않았다.

### 5단계 — 2-물체 모델 (`results/two_object_fit.json`)
`D_px = D_mm/s + c`를 F3 **정상** 시편(NO7·NO8·NO9·NO10·NO1재촬영, 핀 10.000mm + 샤프트
실측mm)에 맞췄다. **NG1·NG2는 적합에 쓰지 않았다**(0.5절 규칙 4 "NG1·NG2를 학습이나
하이퍼파라미터 튜닝에 쓰지 않는다. 평가에만 쓴다" — Codex 3차 adversarial-review 반영,
이전 버전은 NG까지 적합에 넣어 규칙을 어겼었다). NG1·NG2는 이 식으로 **held-out 예측**만
한다(샤프트 px를 mm으로 역산해 실측mm과 비교). NG를 적합에 포함한 예전 결과는
`sensitivity_with_ng`에 참고용으로만 남겼다.

**적합(정상 시편만, descriptive):**

| | s(mm/px) | 부트스트랩 퍼짐(2.5~97.5%ile) | c(px) | s/물리스케일(0.0367) |
|---|---|---|---|---|
| NO7 포함 (n=5묶음) | 0.03648 | [0.03642, 0.03653] | -3.39 | 0.994 |
| NO7 제외 (n=4묶음) | 0.03645 | [0.03641, 0.03649] | -3.60 | 0.993 |

**NG1·NG2 held-out 예측(적합에 전혀 안 씀):**

| NG | 실측mm | 예측mm(NO7 포함 적합) | 오차 | 실측 방향 | 예측 방향 |
|---|---|---|---|---|---|
| NG1 | 19.993 | 20.036 | **+42.6µm** | 하한 아래 | 상한 위 |
| NG2 | 20.002 | 20.049 | **+46.9µm** | 하한 아래 | 상한 위 |

- **held-out으로 보면 이 모델은 못 쓴다.** NG1·NG2는 실측이 하한(20.010) 아래인데, 이
  식은 둘 다 상한(20.030) 위로 예측한다 — **오차 크기(43~47µm)뿐 아니라 방향도 완전히
  틀렸다**(규칙 11). 적합 단계에서 s가 물리 스케일과 가까웠던 것과는 별개로, 샤프트
  개별 시편 간 미세 차이는 이 모델로 설명되지 않는다 — 0.2절의 알려진 문제(px와 실측
  사이 양의 상관 없음, r=-0.35)가 그대로 남아있다는 뜻이다.
- **"s가 물리 스케일과 거의 일치한다"는 검증 근거가 아니라 기술통계다.** 이 적합은
  실질적으로 핀(10.000mm 고정점 하나)과 샤프트 평균(≈20mm 부근 한 덩어리) **두 점으로
  하는 2점 계산**에 가깝다. 부트스트랩이 묶음 ID를 재표집해도 핀 mm은 항상 10.000으로
  고정이고 같은 묶음의 핀·샤프트는 같은 촬영 조건을 공유하므로, "2군집 구조" 자체가 거의
  안 바뀐다 — 그래서 퍼짐이 좁게 나오는 것이지, 독립 재장착·깊이·카메라·초점 불확실성을
  반영한 결과가 아니다(Codex 3차 review 반영, `two_object_fit.json`의 `caveat_on_spread`
  필드에도 같은 내용을 적었다). **그래서 이 수치를 "물리 스케일 검증에 성공했다"는 근거로
  쓰지 않는다** — "핀과 샤프트 평균의 거친 배율이 서로 맞는다" 정도의 서술 통계로만 읽는다.
- **"같은 평면이라면"**: 핀과 샤프트는 깊이·화면 위치가 달라 비텔레센트릭이 아닌 폰 렌즈의
  시차·초점 영향을 다르게 받는다. s가 물리 스케일과 가깝게 나온 것이 "같은 평면" 가정이
  맞아서인지, 핀·샤프트 각각의 치우침이 우연히 비슷한 방향으로 상쇄돼서인지는 이 데이터로
  가를 수 없다.
- (참고) `sensitivity_with_ng`: NG까지 적합에 넣은 예전 방식의 수치도 남겨 뒀다(s=0.03643/
  0.03640, 규칙4 위반이라 결론 근거로 쓰지 않음). 정상만 쓴 적합과 큰 차이는 없다 — NG
  2점이 전체 회귀 기울기에 큰 영향을 못 준다는 뜻이지, NG 포함이 "더 맞다"는 뜻이 아니다.

### 6단계 — 셋업 모니터 (`results/setup_monitor.csv`)
2신호: (a) 핀 px로 `focus_state`("F1F2" 또는 "F3", 269.64~270.62px 사이 270.2px 기준) +
직전 묶음과 `focus_state`가 다른지(`focus_changed`) + 묶음 안 프레임 순서 흐름(묶음7=NO7에서
-0.39px, 묶음11=NG1에서 -0.22px로 경고 임계값 0.2px를 넘는다). (b) 팀원B 고정물(받침대
경계 xL·xR·창아래 yB, 4단계에서 고친 "마지막장->첫장" 기준) 묶음 경계 이동, 1px 임계값
(팀원B 제안과 동일 — 우리 자신의 핀 xL은 4px 격자라 1px 문턱에는 너무 거칠어서 참고용으로만
같이 적었다). **`focus_changed`를 `needs_reshoot`에 포함했다**(Codex 2차 review 반영 —
전에는 `f3_threshold_over`를 계산만 하고 최종 판정에 안 넣어서, 묶음7이 F2->F3로 바뀌는데도
그 신호가 결정에 영향을 안 줬다). 묶음7 `focus_changed=True`, 묶음13 `focus_changed=False`를
`assert`로 고정해 뒀다(스크립트 실행 시 통과 확인). `window_clipped=True`인 프레임(묶음2의
2장)은 `pin_px_mean`·`drift_total_px` 계산에서 뺐다 — 뺀 개수를 `n_window_clipped_excluded`
열에 적었다(묶음2만 2, 나머지는 0). 묶음2는 남은 8장으로 다시 계산해도 판정은 바뀌지 않았다.
- **핀 px 임계값의 한계:** F1과 F2는 핀 px 뭉치가 겹친다(F1 268.95~269.69px, F2
  269.44~269.51px) — `focus_state`는 "F1F2"/"F3" 두 값뿐이고, **실제 묶음4→5의 F1→F2
  전환은 이 신호로 못 잡는다.** 분류표의 "초점" 열이 있어야만 안다.
- **판정: 묶음1~12가 재촬영 경고에 걸리고, 묶음13(NO1 재촬영)만 걸리지 않는다**(`focus_changed`를
  넣기 전과 결과는 같다 — 묶음7이 이미 drift·position 신호로도 걸려 있었기 때문. 다만
  이제는 F3 전환 자체가 판정에 쓰인다는 게 코드로 보장된다).
- 이게 규칙이 느슨하다는 뜻이 아니다. 팀원B 결론 그대로("시편을 바꿀 때마다 카메라가
  흔들렸다") — **이번 10-08 촬영은 거의 매 묶음 전환에서 1px 넘는 고정물 이동이 실제로
  있었다.** 모니터는 핀 px 임계값으로 F1/F2 뭉치 vs F3 뭉치를 완전히 가르고, NO7의 초점
  흐름도 정확히 잡아낸다 — **신호 자체는 쓸 수 있다.** 다만 이 데이터셋 자체가 거의 전부
  "경고 대상"이라, 이번 데이터로는 "좋은 묶음과 나쁜 묶음을 가른다"는 것을 보여주지
  못했다(변별력 검증은 폰을 고정한 다음 촬영에서 다시 봐야 한다).
- 팀원B 제안("시편마다 첫 장의 고정물 위치를 기준과 비교해 1px 넘으면 경고")과 신호(b)는
  같은 발상이다. 다른 점은 팀원B는 기준(묶음3 첫 장)과의 절대 비교이고, 우리는 "직전 묶음
  마지막 장과의 연속 비교"라는 점이다.

## 팀원B 결과와 대조
| 항목 | 일치 여부 |
|---|---|
| 핀 위 경계 기울기 -1.3~-1.7px/100px, 아래 평평 | **대체로 일치**(방향·크기 비슷, 범위는 우리가 약간 넓음) |
| 핀 측정 자리가 실제 오차원, 핀 상대 위치가 낫다 | **일치** |
| 핀 이동 시점: 묶음1·2 사이(LOG70) | **일치**(분류표의 "묶음1 안" 해석과는 불일치) |
| 카메라가 시편 바꿀 때마다 흔들림 | **일치** |
| NO9/NG1 근처 "놓임" 잔차 | **방향 일치**(팀원B는 ±13µm 잔차로, 우리는 픽셀 단위 y 점프로 각자 독립적으로 시편 쪽 원인을 지목) |
| 평균 찍기를 이기는가(참고) | 다루지 않음(E6에서) |
환산 방식: 우리 수치는 모두 cross50 px(핀 비율 보정 없음), 팀원B의 LOO MAE 등은 핀 비율
보정(지름=샤프트px/핀px×10)을 쓴다 — 0.4절에 따라 이 둘을 직접 비교하지 않았다.

## Codex 2차 adversarial-review 반영 표
| 지적 | 반영 여부 | 근거 |
|---|---|---|
| [high] `pin_relative` 창이 검출된 핀 범위(`xR`)를 넘어갈 수 있음 | **반영함** | `measure_pin()`에서 `xb=min(xL+rel_window[1], xR)`로 클램프하고, 잘렸으면 `window_clipped=True`를 반환하도록 고쳤다. `step2_position_compare.py`는 캐시 재집계 대신 `measure_pin()`을 130장에 직접 호출하도록 다시 썼다(런타임 API=평가 산출물). 실제로 클램프에 걸린 건 묶음2의 2장뿐이고, 값은 거의 안 바뀌었다(269.61/269.71px, 동일) — README에 기록. |
| [high] `f3_threshold_over`가 `needs_reshoot` 판정에서 빠짐 | **반영함** | `f3_threshold_over`(bool) -> `focus_state`("F1F2"/"F3")로 바꾸고, 직전 묶음과 다른지 보는 `focus_changed`를 추가해 `needs_reshoot = focus_changed or drift_warn or position_warn`에 포함했다. 묶음7(F2->F3) `focus_changed=True`, 묶음13(F3->F3) `focus_changed=False`를 `assert`로 고정했다. F1->F2 전환은 핀 px 뭉치가 겹쳐 이 신호로 못 잡는다는 한계를 README·docstring에 적었다. |
| [medium] 카메라 "경계 이동"이 묶음 경계가 아니라 첫-첫 비교였음 | **반영함** | `step4_camera_shift.py`를 "이전 묶음 마지막 장 -> 이 묶음 첫 장" 기준으로 다시 계산하도록 고쳤다. 기존 첫-첫 비교는 `*_first_to_first` 열로 분리해 진단용으로 남겼다. `step6_setup_monitor.py`의 docstring이 애초에 "마지막 장 -> 첫 장"이라고 적어 뒀던 것과 이제 코드가 실제로 일치한다. 결론(묶음1~12 경고, 묶음9·11 "시편 놓임" 후보)은 기준을 고쳐도 바뀌지 않았다. |

## Codex 3차 adversarial-review 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] NG1·NG2가 2-물체 적합에 섞여 들어감 | **맞음** | **반영함** | `docs/EXPERIMENTS_20261008.md` 0.5절 규칙4 "NG1·NG2를 학습이나 하이퍼파라미터 튜닝에 쓰지 않는다. 평가에만 쓴다"를 인용 확인. `step5_two_object_fit.py`를 정상 F3 시편(NO7~10, NO1재촬영)만으로 적합하고, NG1·NG2는 그 식으로 held-out 예측만 하도록 고쳤다. held-out 결과 NG1·NG2는 오차 43~47µm에 **방향까지 틀렸다**(실측은 하한 아래인데 예측은 상한 위) — 적합에 NG를 섞었을 때보다 훨씬 더 정직한 결과다. NG 포함 예전 수치는 `sensitivity_with_ng`에 참고용으로만 남겼다. |
| [high] 부트스트랩 CI가 2점 계산의 불확실성을 가림 | **맞음**(코드로 확인: 묶음을 재표집해도 핀 mm은 항상 10.000 고정, 묶음별 핀·샤프트는 같은 촬영 조건 공유) | **반영함** | JSON 필드명을 `s_ci95`/`c_ci95`(95% 신뢰구간처럼 읽힘)에서 `s_bootstrap_spread_p2_5_p97_5`로 바꾸고 `caveat_on_spread` 필드로 한계를 명시했다. README에서 "s가 물리 스케일과 거의 일치한다"는 문장을 "검증 근거"가 아니라 기술통계로 다시 썼고, held-out NG 예측이 완전히 틀렸다는 사실을 함께 적어 과신을 막았다. |
| [medium] `window_clipped`가 하류에 전파 안 됨 + xR 정의 불일치(scan_pins vs measure_pin, 130장 전부 2px 차이) | **맞음**(코드로 재현: `measure_pin_profile`의 `x_to`=9748, `x_per_col[-1]`=9746, 모든 이미지에서 2px 차이) | **반영함** | `scan_pins.py`의 xL/xR을 `x_per_col[0]/[-1]`(마지막 "측정한" 열) 대신 `x_from`/`x_to`(coarse_section의 원시 경계, `measure_pin()`과 같은 정의)로 통일했다. 왜 이쪽인지: `measure_pin()`의 클램프 로직(`xb=min(xb_requested, xR)`)이 이 원시 경계를 기준으로 삼기 때문에, `window_clipped` 판정과 같은 좌표계를 써야 앞뒤가 맞는다. `scan_pins.py`에 `measure_pin()`의 xL/xR과 다르면 바로 멈추는 교차 검사를 추가했다(130장 전부 통과 확인). `step3`(열 추가)·`step5`(적합에서 clipped 프레임 제외)·`step6`(평균·흐름 계산에서 clipped 프레임 제외)에 `window_clipped`를 전파했다. |
| [medium] `measure_pin()` 기본값 변경이 조용한 의미 변경 | 맞지만 **지시서(0.4절, E5 방법2 "선정 방식을 기본값으로")와 충돌** — 사용자 확인 후 기본값은 유지하고 명시적 API로 보완하기로 결정 | **부분 반영** | 사용자에게 확인: 기본값을 "auto"로 되돌리면 지시서가 요구하는 "E5 선정 방식을 기본값으로" 조건을 어기고, 이후 E4·E6이 `measure_pin(gray)`를 그냥 부를 때 조용히 구식 auto를 쓰게 되는 **더 나쁜 조용한 불일치**가 생긴다고 판단해 기본값(pin_relative)은 유지하기로 결정. 대신 `measure_pin_auto()`/`measure_pin_relative()` 명시적 래퍼를 추가하고, `E1_module/reproduce.py`가 `measure_pin_auto()`를 쓰도록 바꿨다. 두 래퍼가 각각의 `position_method` 명시 호출과 같은 값을 내고, `measure_pin()` 기본값이 실제로 `measure_pin_relative()`와 같은지 확인하는 회귀 테스트를 `E1_module/reproduce.py`에 추가했다(`check_measure_pin_default_consistency`, 3장 샘플 확인, exit 0). `measure_pin()` docstring 맨 앞에 "기본값이 바뀌었다" 경고와 두 래퍼 사용법을 적었다. |

## Codex 4차 review(`/codex:review`) 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] `experiments/1st` 범위 밖 파일이 패치에 섞여 있음(`experiments/1st/박스 비교 실험/compare_roi.py`) | **틀림 — 반려** | 반려함 | 이 파일은 내가 만들거나 건드린 적이 없다. 세션 시작 전부터 untracked 상태로 있던 파일이고(`git status --short`로 확인, staged diff에도 전혀 없음), 0.5절 규칙 9(`experiments/1st`는 건드리지 않는다)를 지켜 계속 손대지 않았다. Codex가 "현재 변경"을 `git status --short --untracked-files=all` 전체로 스캔하면서 내 패치와 무관한 기존 untracked 파일까지 같이 집어낸 것으로 보인다. |
| [P2] `measure_pin()` 기본값 일관성 체크가 우연히 같은 값이면 실패 처리함 | **맞음** | **반영함** | `check_measure_pin_default_consistency()`에서 "default와 auto의 `diameter_px`가 같으면 실패"로 보던 조건을 지웠다 — 두 창이 다른 값을 내는 게 기본값이 맞다는 증거가 아니라, 숫자가 같을 수도 있다. 대신 `measure_pin()`이 실제로 반환하는 `position_method` 메타데이터(`default["position_method"] == "pin_relative"`)로 확인하도록 바꿨다. 재실행해서 PASS 확인. |
| [P2] `step4`의 `groupby().first()/.last()`가 열마다 따로 첫/마지막 non-NA를 골라 다른 프레임 값을 한 행에 섞을 수 있음 | **맞음**(일반적으로; 이번 데이터는 관련 열에 NaN이 전혀 없어 수치 자체는 안 바뀜, 코드로 확인) | **반영함** | `first_frame_per_bundle`/`last_frame_per_bundle`을 `groupby().nth(0)`/`nth(-1)`로 바꿨다 — 그룹 안 "몇 번째 행 전체"를 그대로 고르므로 같은 프레임의 값만 쓰인다(열별로 따로 안 고름). 재실행 결과 수치는 기존과 동일(이번 데이터에 NaN이 없어서), 다음 촬영에서 일부 열이 비어도 안전하다. **추가 검증(정렬·인덱스 재확인 요청 반영):** 코드가 `nth()` 전에 반드시 `sort_values(time_col)`을 거치는지 확인했고, 실제 데이터를 행 순서를 무작위로 섞어 넣어도(`sample(frac=1)`) `nth(0)`/`nth(-1)` 결과가 섞기 전과 완전히 똑같음을 확인했다(순서 무관성 확인) — `시각`(ours) 열과 `file`(팀원B, `YYYYMMDD_HHMMSS.jpg` 형식이라 문자열 정렬=시간순)이 모두 올바르게 정렬된다. pandas 3.0.6에서 `nth()`가 원본 행 인덱스를 유지한 채 반환하는 것도 확인했고, 코드가 그 직후 `.set_index("묶음")`으로 그룹 키를 인덱스로 바꿔 두므로 이어지는 `.join()`이 올바르게 정렬된다(합성 예제로 join 결과까지 확인). |

## 그림 (한글 라벨, Codex 2차 리뷰 추가 요청 반영)
- `figures/pin_width_profile.png`: s-구간별 핀 폭 프로파일(cross50/grad_peak, 팀원B 인용값과 겹쳐 그림).
- `figures/pin_position_methods.png`: 묶음별 세 측정 위치 방식의 평균±sd 비교.
- 두 그림 모두 전에는 한글 라벨이 □로 깨졌다(DejaVu Sans에 한글 글리프가 없음). `shaftlab/plotting.py`의
  `setup_korean_font()`로 `AppleGothic`(이 Mac에 설치돼 있어 1순위로 선택됨)을 `rcParams`에
  설정한 뒤 다시 그려서 직접 열어 깨짐이 없음을 확인했다. 숫자는 바꾸지 않았다(폰트만 변경).
  두 파일 모두 1MB 이하(각 83KB, 91KB).

## E1 영향 (`shaftlab.measure_pin` 기본값 변경)
`measure_pin()`의 기본값을 `position_method="pin_relative"`(s=60~105)로 바꿨다(이전에는
암묵적으로 "auto"였다). **E1이 재현한 `pin_gauge_raw.csv`는 원본 `eda_pin_gauge.py`의 자동
구간으로 만들어졌으므로, `E1_module/reproduce.py`는 이제 명시적 래퍼 `measure_pin_auto()`를
써서 그 재현 검증을 그대로 PASS 유지한다**(재확인함, 아래 "실행 명령" 참고). 다만 **기본값으로
`measure_pin(gray)`를 그냥 부르면 이제 다른 값이 나온다** — 묶음 평균 기준 차이는
0.015~0.167px(아래 표), `results/pin_position_compare.csv`의 `method_a_auto` 열과
`method_c_pin_relative` 열 차이와 같다.

**명시적 API 두 개를 추가했다(Codex 3차 adversarial-review 반영):**
- `measure_pin_auto(gray_full, **kwargs)` == `measure_pin(gray_full, position_method="auto", **kwargs)`
- `measure_pin_relative(gray_full, **kwargs)` == `measure_pin(gray_full, position_method="pin_relative", **kwargs)`(기본값과 같음)

호출부가 기본값에 기대지 않고 어느 방식을 쓰는지 이름으로 분명히 할 수 있다. `E1_module/reproduce.py`에
`check_measure_pin_default_consistency()`를 추가해 (1) `measure_pin(gray)`가 `measure_pin_relative(gray)`와
같은 값을 내는지(기본값이 실제로 pin_relative인지), (2) `measure_pin_auto(gray)`가
`measure_pin(gray, position_method="auto")`와 같은 값을 내는지를 샘플 3장에서 확인하는
회귀 테스트를 넣었다(재실행해서 통과 확인). `measure_pin()` docstring 맨 앞에 기본값이
바뀐 사실과 두 래퍼의 차이를 명시했다.

| 묶음 | (a) 기존 자동 | (c) 새 기본값 | 차이 |
|---|---|---|---|
| 1 | 269.449 | 269.282 | 0.167 |
| 5 | 269.587 | 269.442 | 0.145 |
| 7 | 270.921 | 270.773 | 0.148 |
| 11 | 270.702 | 270.687 | 0.015 |
(전체 13묶음은 `results/pin_position_compare.csv` 참고. 최대 0.167px, 최소 0.015px.)

`measure_section`은 수정하지 않았다는 E1의 전제도 지켰다 — `_measure_window`로 로직을
뽑아내는 리팩토링만 했고(산술 동일), E1의 `reproduce.py`를 재실행해 베어링·핀 모두
`max|Δ| < 1e-11px`로 그대로 PASS함을 재확인했다.

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음** — 이 데이터로 최종 성능을 주장하지 않는다(규칙 8).
- 1단계 기울기 재현은 방향·크기가 "대체로" 맞을 뿐 완전히 같지는 않다(측정 window·임계값
  구현 차이). 다음 촬영에서 같은 핀으로 다시 확인한다.
- 5단계 2-물체 적합은 사실상 2점(핀/샤프트 평균) 계산이라 "같은 평면" 가정을 이 데이터로
  검증할 수 없다. 다음 촬영에서 핀을 여러 깊이·위치에 둬서 확인해야 한다.
- 5단계 NG held-out 평가는 NG가 2개뿐이라(NG1, NG2) 일반화할 수 없다. 둘 다 같은 방향으로
  틀렸다는 것(상한 위로 예측)이 우연인지 체계적인지는 NG 시편이 더 있어야 안다.
- 6단계 셋업 모니터는 "신호가 반응하는가"만 확인했다. "좋은/나쁜 묶음을 가르는 변별력"은
  폰을 고정한 안정적인 촬영에서 다시 봐야 한다(이번 데이터는 거의 전부 경고 대상이라 변별력
  검증이 안 된다).
- 묶음9/10/11 근처 "시편 놓임 후보" 판정은 가설이다. 확정은 같은 시편을 여러 번 다시 끼워
  촬영해서 가린다(0.5절 1011_camstate README의 제안과 같음).

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate   # 없으면: python3.11 -m venv .venv && pip install -r requirements.txt
cd E5_pin_gauge
python3 scan_pins.py              # results/pin_profile.csv, cache/pin_bearing_scalars.csv
python3 step1_profile_check.py    # results/pin_slope_by_bundle.csv, figures/pin_width_profile.png
python3 step2_position_compare.py # results/pin_position_compare.csv, figures/pin_position_methods.png
python3 step3_pin_state.py        # results/pin_state.csv
python3 step4_camera_shift.py     # results/camera_shift_table.csv
python3 step5_two_object_fit.py   # results/two_object_fit.json
python3 step6_setup_monitor.py    # results/setup_monitor.csv

# E1 영향 재확인(shaftlab 변경 후 베어링·핀 재현이 여전히 PASS인지)
cd ../E1_module && python3 reproduce.py
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, matplotlib 3.11.2
- torch: 사용하지 않음
- 랜덤 시드: 5단계 부트스트랩 `seed=20261013`(`step5_two_object_fit.py`의 `SEED`)

## 변경 파일 목록
```
M experiments/2nd/requirements.txt           (matplotlib 추가)
M experiments/2nd/shaftlab/__init__.py       (measure_pin_profile, measure_pin_auto, measure_pin_relative export)
M experiments/2nd/shaftlab/measure.py        (measure_pin_profile 추가, _measure_window 리팩토링,
                                               measure_pin에 position_method 추가 + 기본값 변경 +
                                               pin_relative 창 xR 클램프 + window_clipped 플래그 +
                                               measure_pin_auto/measure_pin_relative 명시적 래퍼 추가)
A experiments/2nd/shaftlab/plotting.py       (setup_korean_font())
M experiments/2nd/E1_module/reproduce.py     (measure_pin_auto() 사용, 기본값 일관성 회귀 테스트 추가,
                                               해당 테스트는 position_method 메타데이터로 확인)
A experiments/2nd/E5_pin_gauge/README.md
A experiments/2nd/E5_pin_gauge/scan_pins.py               (xL/xR을 x_from/x_to로 통일 + 교차 검사)
A experiments/2nd/E5_pin_gauge/step1_profile_check.py
A experiments/2nd/E5_pin_gauge/step2_position_compare.py  (measure_pin 직접 호출로 재작성)
A experiments/2nd/E5_pin_gauge/step3_pin_state.py         (window_clipped 열 추가)
A experiments/2nd/E5_pin_gauge/step4_camera_shift.py      (마지막장->첫장 기준 + first_to_first 진단열 +
                                                             groupby.nth()로 열별 행 뒤섞임 방지)
A experiments/2nd/E5_pin_gauge/step5_two_object_fit.py    (NG 제외 적합 + held-out 평가 + sensitivity_with_ng + CI를 기술통계로 재표기)
A experiments/2nd/E5_pin_gauge/step6_setup_monitor.py     (focus_state/focus_changed + assert + window_clipped 제외)
A experiments/2nd/E5_pin_gauge/results/pin_profile.csv
A experiments/2nd/E5_pin_gauge/results/pin_slope_by_bundle.csv
A experiments/2nd/E5_pin_gauge/results/pin_position_compare.csv
A experiments/2nd/E5_pin_gauge/results/pin_state.csv
A experiments/2nd/E5_pin_gauge/results/camera_shift_table.csv
A experiments/2nd/E5_pin_gauge/results/two_object_fit.json
A experiments/2nd/E5_pin_gauge/results/setup_monitor.csv
A experiments/2nd/E5_pin_gauge/figures/pin_width_profile.png       (재생성, 한글 폰트)
A experiments/2nd/E5_pin_gauge/figures/pin_position_methods.png    (재생성, 한글 폰트)
(재실행으로 변경되었으나 내용은 동일: experiments/2nd/E1_module/results/reproduce_diff.csv)
```
(`experiments/2nd/cache/pin_bearing_scalars.csv`는 `.gitignore`의 `**/cache/`에 걸려
스테이징 대상 아님.) 5MB 초과 파일 없음(`pin_profile.csv`가 가장 커서 약 1.9MB, 그림 두 장은
각 1MB 이하).

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- `measure_pin()` 기본값을 "auto"로 되돌리지 않고 pin_relative로 유지한 채 `measure_pin_auto()`/
  `measure_pin_relative()` 명시적 래퍼만 추가하는 쪽으로 결정했다(지시서의 "선정 방식을
  기본값으로" 요구와 충돌해 사용자 확인 후 결정). 이 절충이 적절한지, 아니면 그래도 기본값
  자체를 더 보수적으로(예: 둘 다 位치 인자를 필수로 받게) 바꿔야 하는지 봐 달라.
- step5의 NG held-out 평가가 NG 2개뿐이라는 한계를 README·한계 절에 적었지만, "둘 다 같은
  방향(상한 위)으로 틀렸다"는 관찰을 과도하게 해석(예: "체계적 편향이 있다"는 식으로)하지
  않았는지 봐 달라.
- `window_clipped` 제외 로직(step5·6)이 "제외 후 표본이 너무 적으면 전체로 되돌린다"는
  안전장치를 뒀는데(`step6`의 `len(d_clean) < 3` 분기), 이 임계값(3장)이 임의적이다 — 더
  나은 기준이 있는지, 아니면 이번 데이터(최대 제외 2장)에서는 상관없는지 봐 달라.
