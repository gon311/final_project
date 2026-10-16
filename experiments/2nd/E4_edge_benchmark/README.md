# E4. 엣지 정의 벤치마크

## 한 줄 결론
**현재 촬영 조건(13셋업, 10장 반복 프레임, 재장착 반복 없음 — 규칙 8) 안에서는** cross50(1순위,
기존 EDA와 동일) + moment_centroid(2순위, 계산이 가벼운 대안)가 가장 안정적인 teacher 후보다
— 둘 다 **"가장 강한 하단 그래디언트가 있는 단"**(E3 번호로 거의 항상 2단, 물리적으로 맨 바깥
단이 아닐 수 있다)을 따라가고, 셋업 내 σ·열 잔차 σ가 가장 좋다. grad_peak는 NO4에서 +5.7px(B단)
튀는 게 팀원A V3 엔진에서도 그대로 재현됐고, ESF 2·3단은 열마다 적합이 "진짜 단 붕괴"로 흔들려
(`esf3_outer`는 13셋업 중 5개에서만 적합 자체가 성립한다) teacher로 쓰기엔 너무 불안정하다.
**실측 상관·MAE로 고르지 않았다**(완료 기준) — 오히려 esf2_outer가 작은 부분표본에서 r=0.84
(p=0.07)로 튀는 걸 직접 확인했고, 이게 "정답을 보고 고르면 안 되는" 함정임을 보여 주는 사례로
남겼다(아래 참고). **단, 이 결론의 범위는 "현 촬영 조건 내 반복성"이다** — 독립 재장착·광학계
변동까지 검증한 건 아니므로 최종 teacher 확정이 아니라 다음 단계(E7 프로토타입)로 넘길
**후보**로만 쓴다(Codex adversarial-review 반영, 아래 "Codex 리뷰 반영 표" 참고).

## 한 일
1. `shaftlab/edge_methods.py`(신규): `edge_subpx`(E1, 원본 그대로)가 이미 주는 cross50·
   grad_peak 외에 세 가지를 추가했다 — `level_crossing`(임의 레벨 교차, 하단 문턱 sweep용.
   `level=0.5`로 부르면 `edge_subpx`의 cross50과 값이 같다, 코드로 확인함),
   `moment_centroid`(그래디언트 1차 모멘트 기반 서브픽셀), `esf_fit`(n=1\~3단 로지스틱(ESF)
   모델을 scipy 최소제곱으로 맞춤, 초기값은 `shaftlab.multistep.find_peaks_subpixel`(E3,
   seg[k]>0 필터 적용된 버전)이 찾은 극댓값을 쓴다). **`esf_fit`은 scipy `least_squares`의
   `success`(수렴 종료 여부)만으로는 못 거르는 "성공적으로 수렴한 퇴화해"(n개 단 중 둘 이상이
   거의 같은 위치로 겹치는 경우)도 `MIN_STEP_SEP=1.5px` 미만 간격 검사로 잡아 실패 처리한다**
   (Codex adversarial-review 반영, 아래 "Codex 리뷰 반영 표" 참고). `measure_section`/
   `_measure_window`/`edge_subpx`는 건드리지 않았다.
2. `shaftlab/paths.py`에 `REFERENCE_BOX = (7844, 3468, 8089, 4468)` 상수 추가(팀원A V3
   엔진 호출용, 0.2절 문서화된 값 그대로).
3. `requirements.txt`에 `scipy==1.17.1` 추가(ESF 최소제곱 적합에 필요, 설치 전 알림 — 규칙
   0.5.7). 설치 외에는 기존 패키지 변경 없음.
4. `scan_definitions.py`: 130장을 디코딩해 베어링부 상·하단(기본 `--step 10`, `--step 2`로도
   돌릴 수 있다 — "한계"·"Codex 리뷰 반영 표" 참고)과 핀 상·하단(E5 `pin_relative` 창,
   `step=2` 고정)에서 cross50/grad_peak/moment/ESF1\~3/하단 문턱 9단계를 모두 뽑는다.
   **ESF 적합 실패(`*_ok=False`)는 fallback값을 쓰지 않고 NaN으로 남긴다**(Codex
   adversarial-review 반영). 원자료는 `cache/e4/bearing_columns{,_step2}.csv`,
   `cache/e4/pin_columns.csv`(커밋 안 함, 0.6절).
5. `step1_summary.py` \~ `step8_step_size_check.py`: 아래 "결과 요약" 참고.
6. **cross50·grad_peak(`edge_subpx` 호출)의 탐색 창을 `measure.py` `_measure_window`와 같은
   `EDGE_SUBPX_WIN=20`으로 좁혔다**(Codex `/codex:review` 1차 반영) — moment·ESF·하단 문턱
   sweep은 다단을 보려고 여전히 `MULTISTEP_WIN=45`를 쓴다. `step3_pin_f3_range.py`도 ESF1
   적합 실패 열을 명시적으로 걸러 `f3_n_valid_images`로 남기도록 고쳤다(같은 라운드 반영).
7. **`esf_fit`이 방향성 극댓값을 `n_steps`개 미만(0개 포함)밖에 못 찾으면 `least_squares`를
   돌리지 않고 바로 실패 처리하도록 고쳤다**(Codex `/codex:review` 2차 반영, 아래 "Codex
   리뷰 반영 표" 참고) — `n_steps=1`일 때는 단 붕괴 검사(`min_gap`)가 비교할 상대가 없어
   무의미해지는데, 전에는 이 경우도 가짜 초기값으로 fitting을 돌려 scipy가 "성공"으로
   잘못 보고할 수 있었다.

## 방법론 메모
- **cross50·grad_peak는 `EDGE_SUBPX_WIN=20`(E1과 동일), moment·ESF·문턱 sweep은
  `MULTISTEP_WIN=45`를 쓴다.** 처음엔 전부 45였는데, `edge_subpx`는 `[center-win,
  center+win]` 안의 가장 강한 그래디언트를 고르므로 창을 넓히면 E1이 원래 고르던 엣지와
  다른 지점(더 멀리 있는 반사띠 등)을 고를 위험이 있다는 지적을 받았다(Codex `/codex:review`).
  **직접 재보니 이번 데이터에서는 차이가 없었다** — 베어링부 열 15×130장 전부에서
  `win=20`과 `win=45`의 cross50/grad_peak 값이 1e-6px 안으로 완전히 같았다(단일 열·전체
  둘 다 확인). 이유는 베어링 하단의 실제 다단 간격이 5\~13px(0.2b)로 ±20px 창 안에 이미
  다 들어와서, 창을 더 넓혀도 "가장 강한" 지점이 바뀌지 않기 때문으로 보인다. **그래도
  고친 게 맞다** — 코드가 "데이터가 마침 이래서" 우연히 맞았던 것이지, 원래는 더 멀리 있는
  반사·특징을 집어올 수 있는 구조였다(핀처럼 다른 섹션, 또는 다음 촬영의 다른 조명에서는
  실제로 달라질 수 있다). 수정 후 전체 파이프라인을 다시 돌렸고, 모든 결과 수치는 이전과
  **동일하게 재현됐다**(아래 결과 표 전부 불변 — 코드 안전성만 개선, 결론 변경 없음).
- **ESF n단의 "위치"는 바깥 단(outer, 배경에 가장 가까운 단)으로 정의했다** — 팀원B
  M2\~M5("바깥 단" 계열)와 같은 방향이라 직접 비교가 된다. 안쪽 단을 쓰는 대안은 README
  "한계"에 남겨 뒀다.
- **문턱 sweep의 "단 전환 비율"은 10%·90%(양 끝)가 아니라 40%·60%(50% 바로 옆)를 비교해서
  정의했다.** 처음엔 10% vs 90%로 했더니 모든 셋업이 1.000(항상 전환)이 나왔는데, 이는
  양 끝 문턱이 애초에 다른 물리적 단 근처에 있어서 당연한 결과였을 뿐 "불안정하다"는 의미가
  아니었다(구현 실수를 스스로 발견하고 고침, 아래 "한계"에도 기록). 50% 바로 옆(실제 측정이
  동작하는 지점)에서 비교해야 "흔들림"을 재는 것이 된다.
- **베어링부 스캔은 기본 `step=10`(열 10px 간격)을 썼다**(E1의 `step=2`보다 성글다) — ESF
  비선형 적합이 열마다 scipy solver를 돌려서 느리기 때문이다. 처음엔 cross50만 `eda_measure_raw.csv`
  (E1이 재현한 원본, step=2)의 `B_d_cross_median` 셋업 σ와 대조했는데(σ_med=0.1221px(NO7
  포함)/0.1241px(제외) vs step=10의 0.1215/0.1223 — 거의 같음), **cross50만 검증한 건
  moment·ESF까지의 순위를 보장하지 못한다는 지적을 받고(Codex adversarial-review) 130장
  전체를 `step=2`로 다시 스캔해 6개 정의 전부를 비교했다**(`step8_step_size_check.py`,
  `results/step_size_sensitivity.csv`). **cross50·grad_peak·moment의 σ는 step=10/step=2
  사이 차이가 0.0006\~0.0014px뿐이고, 안정성 순위(esf1<cross50<moment<grad_peak<esf3_outer
  <esf2_outer)가 두 step 크기에서 완전히 같다.** `step=10`이 teacher 후보 순위를 왜곡하지
  않는다는 직접적인 근거다(esf2_outer만 σ가 0.158→0.228로 더 벌어지는데, 애초에 이미
  "못 쓴다"고 결론 낸 정의라 이 결론에는 영향이 없다).
- **ESF 적합의 "성공" 여부를 scipy `success`만으로 판단하면 안 된다는 걸 직접 확인했다**
  (Codex adversarial-review 반영). 처음엔 `success=False`가 **한 번도** 안 나왔다(전체
  1890열 중 실패율 0%) — scipy는 "정상 종료"라고 보고하는데, 실제로는 n=3인데 진짜 단이
  1개뿐인 열에서 3개 중 2개가 거의 같은 위치(간격 <0.01px)로 겹치고 나머지 1개가 창 끝으로
  멀리 밀려나는 "성공적으로 수렴한 퇴화해"가 흔했다. 적합된 단 사이 최소 간격이 1.5px
  미만이면 실패로 처리하도록 고친 뒤(`MIN_STEP_SEP`, `shaftlab/edge_methods.py`)에야
  `esf2_outer`·`esf3_outer`의 진짜 실패율(각각 67%·81%, 묶음별로는 0\~100%)이 드러났다 —
  아래 1단계 참고.

## 결과 요약

### 1단계 — 정의별 안정성 (`results/edge_method_benchmark.csv`, `results/esf_fit_failure_rate.csv`, `figures/definitions_by_setup.png`)
| 정의 | 유효 셋업 수(/13) | 셋업 내 σ(px) | σ(NO7 제외, px) | 열 잔차 σ(px) | cross50 대비 체계차(px) |
|---|---|---|---|---|---|
| **cross50** | 13 | 0.1215 | 0.1223 | **0.1176** | 0 (기준) |
| **esf1** | 13 | **0.1183** | **0.1133** | 0.1303 | −0.091 |
| moment | 13 | 0.1238 | 0.1233 | 0.1566 | −0.004 |
| grad_peak | 13 | 0.1449 | 0.1659 | 0.1262 | **+0.423** |
| esf2_outer | **6** | 0.1575 | 0.1575 | 0.3802 | **+2.777** |
| esf3_outer | **5** | 0.1528 | 0.1528 | **0.2951** | **+4.608** |

- **cross50과 esf1이 가장 안정적이다.** cross50은 열 잔차가 가장 작고(0.118px), esf1은
  셋업 내 σ가 가장 작다(0.113px, NO7 제외) — 사실상 동점권이고 같은 단(가장 강한 그래디언트)
  을 따른다(둘이 서로 −0.09px 체계차밖에 안 난다).
- **grad_peak는 그림(`definitions_by_setup.png`)에서 묶음4(NO4)에만 +5.7px 튄다** — 0.2절에
  이미 알려진 "V3가 NO4에서 B단을 잡아 튀었다"는 사실이 우리 데이터·팀원A 엔진 둘 다에서
  재현된다(아래 "참조 대조" 참고). NO7 제외 시 σ가 오히려 더 나빠지는(0.145→0.166) 유일한
  정의이기도 하다 — 재초점 흐름(NO7)보다 다른 프레임의 변동이 더 크다는 뜻.
- **esf2_outer·esf3_outer는 "불안정하다"를 넘어 대부분 셋업에서 아예 못 쓴다.** `MIN_STEP_SEP`
  단 붕괴 검사를 넣은 뒤(위 "방법론 메모") 재보니, **ESF3는 13셋업 중 5개(묶음1\~6 중 일부)
  에서만 적합이 성립하고, F3 초점 구간(묶음7\~10·12)에서는 실패율이 100%다**
  (`results/esf_fit_failure_rate.csv`). 성립하는 5\~6개 셋업만 놓고 봐도 열 잔차 σ가
  esf2 0.38px·esf3 0.30px로 cross50(0.12px)의 2.5\~3배, cross50 대비 체계차는 esf2 +2.8px,
  esf3 +4.6px — **이전 라운드(단 붕괴 검사 전)보다 수치가 더 나빠졌다**, 그만큼 전에는
  퇴화해가 섞여 불안정성을 과소평가했었다는 뜻이다. E3가 이미 본 "핀도 99%가 3개 잡힌다" 문제의 ESF 버전이자, 그보다
  더 직접적으로 "진짜 단이 없으면 적합 자체가 실패한다"는 증거다.

### 2단계 — 하단 문턱 sweep + 단 전환 비율 (`results/bottom_threshold_sweep.csv`, `results/bottom_step_transition_by_setup.csv`, `figures/bottom_threshold_sweep.png`)
- 상단 50% 고정, 하단만 10\~90%로 바꾸면 **셋업 중앙값 지름이 10%에서 90%까지 약 +14px
  오른다**(그림) — "하단 문턱이 결과를 가장 크게 흔드는 축"(0.2b)이 그대로 재현된다.
- 정상 10개 셋업의 Spearman r(문턱별, 실측 대비): **−0.417\~+0.448**(순열 p 전부 >0.2, 유의한
  문턱 없음) — 팀원B의 81조합 결과(−0.47\~+0.59, 최선조차 p=0.19)와 **같은 자릿수·같은 결론**
  ("문턱을 어디로 잡아도 우연 수준 이상의 신호가 없다").
- **단 전환 비율("단 후보"로 ESF3 inner/mid/outer를 쓴 지표)은 ESF3 단 붕괴 검사를 넣은 뒤
  결론이 바뀌었다.** 전에는 (퇴화해 포함) 0.000\~0.993으로 셋업마다 극단적으로 갈린다고
  봤는데, **퇴화한 ESF3 적합을 "단 후보"에서 뺀 뒤에는 묶음7\~10·12·13이 통째로 계산 불가가
  되고(ESF3 실패율 100%), 적합이 성립하는 6개 셋업(묶음1\~6)에서는 40%↔60% 전환 비율이
  **전부 0.000**이 나온다**(`results/bottom_step_transition_by_setup.csv`). 해석: 진짜로
  단이 분리돼 있으면(ESF3가 성립할 정도로) 40\~60%의 좁은 흔들림 폭(약 3.5px, 실제 단 간격
  5\~10px보다 작다)으로는 "가장 가까운 단"이 바뀌지 않는다 — **전에 본 넓은 분산은 이 흔들림
  자체의 성질이 아니라, 신뢰할 수 없는 ESF3 "단 후보" 위치가 섞여 들어간 결과였다.** 이
  지표를 이 형태로 쓰는 한 "셋업마다 단 전환이 얼마나 다른가"에 답을 못 준다 — 아래 "한계"에
  정직하게 남긴다.
- **이 sweep으로 teacher를 고르지 않는다**(완료 기준) — Spearman r 수치는 참고로만 둔다.

### 3단계 — Ø10 핀 px의 F3 내 범위 (`results/pin_f3_range.csv`)
| 정의 | F3 유효 이미지(/70) | 범위(px) | σ(px) |
|---|---|---|---|
| moment | 70 | **0.505** | **0.110** |
| cross50 | 70 | 0.524 | 0.128 |
| esf1 | 70 | 0.531 | 0.106 |
| grad_peak | 70 | **0.733** | 0.185 |
- 핀은 F3 구간 안에서 물리적으로 고정돼 있어야 하는데, **grad_peak만 범위·σ 모두 뚜렷이
  크다** — 1단계(베어링부)에서 본 grad_peak 불안정과 같은 방향. moment·esf1·cross50은
  셋 다 비슷하게 작다.
- **"유효 이미지" 열을 추가했다**(Codex `/codex:review` 반영) — 전에는 ESF1 적합이 실패한
  열이 있어도 `median`/`max`/`min`이 조용히 NaN을 건너뛰면서 "F3 70장"이라는 분모만
  그대로 찍혔다. 이번 데이터에서는 네 정의 전부 70/70으로 실제 실패가 없었지만, 앞으로
  실패가 생기면 이 열이 바로 드러낸다.

### 4단계 — (참고) Spearman + 순열 p (`results/spearman_by_definition.csv`)
**teacher 선택에 쓰지 않는다**(완료 기준). 그래도 적어 두는 이유: `esf2_outer`가 (ESF3 적합이
성립하는 6개 셋업만 놓고) r=**0.837**(순열 p=**0.067**)로 다른 정의보다 높게 나왔다. **이걸
teacher로 고르면 안 된다** — (1) ESF3 단 붕괴 검사를 넣기 전에는 이 수치가 r=0.859, p=0.002
(6개 정의 Bonferroni 0.05/6≈0.0083보다도 작은, 겉보기엔 "확실히 유의"한 값)이었는데, 실패한
적합을 제대로 빼고 나니 **똑같은 정의의 같은 종류 수치가 p=0.067(유의하지 않음)로 내려갔다**
— 데이터 품질 문제가 섞인 상관은 이렇게 쉽게 사라진다. (2) 1단계에서 이미 esf2_outer가 13
셋업 중 6개에서만 적합이 되고, 그 6개에서도 열 잔차 σ(0.380)가 cross50(0.118)의 3배라고
확인했다 — "불안정한 지표가 작은(그것도 자기 유리한 쪽으로 편향된) 부분표본에서 우연히
실측과 잘 맞은" 사례다. 팀원B가 0.2b에서 본 "81조합 중 정답 보고 고른 최선도 p=0.19"와
정확히 같은 함정이고, 우리는 6개만 비교했는데도 두 번(단 붕괴 검사 전후 모두) 비슷하게 걸렸다
— **이 지표군 자체가 과신하기 매우 쉽다는 걸 두 라운드 연속으로 보여준 셈이다.**

### 5단계 — (참고) 평균 찍기 대비 LOO MAE (`results/loo_mae_by_definition.csv`)
| 정의 | 환산 | n_setups | LOO MAE(µm) | LOO 최대(µm) |
|---|---|---|---|---|
| **(기준) 평균 찍기** | − | 10 | **5.78** | 12.67 |
| cross50 | 물리스케일 | 10 | 25.99 | 67.04 |
| esf1 | 물리스케일 | 10 | 28.28 | 67.89 |
| moment | 핀비율 | 10 | 33.14 | 68.70 |
| esf3_outer | 물리스케일 | **5** | 23.24 | 58.11 |
- **평균 찍기를 이기는 정의가 없다**(E6 전 미리보기, 규칙 10) — 이 수치로 teacher를 고르지
  않는다(완료 기준). 정식 비교는 E6에서 한다.
- **esf2/esf3_outer의 `n_setups`가 10보다 작다는 점을 반드시 같이 읽어야 한다**(Codex
  adversarial-review 반영) — ESF 적합이 통째로 실패하는 셋업은 LOO에서 완전히 빠지므로, 예를
  들어 `esf3_outer`의 23.24µm은 cross50의 25.99µm(n=10)보다 작아 보이지만 **서로 다른,
  "더 쉬운"(ESF가 다단을 찾아낼 만큼 실측 분산이 작은) 5개 셋업 부분집합**에서 계산된 값이라
  직접 비교하면 안 된다. 표에 `n_setups`를 추가해 이 함정을 바로 눈에 띄게 했다.

### 6\~7단계 — 참조 대조 (`results/reference_compare.csv`)
- **팀원A V3 엔진(같은 ROI `REFERENCE_BOX`, 엔진 수정 안 함)과 우리 grad_peak가 묶음4(NO4)
  에서 같이 튄다**: 우리 grad_peak 550.30px, V3 측정값 550.37px(둘 다 다른 묶음의 \~544.7px
  대비 +5.6px) — **"V3가 NO4에서 B단을 잡아 튀었다"는 0.2절 기록이 우리 grad_peak로도,
  팀원A 엔진 자체 재실행으로도 동시에 재현된다.** 같은 현상을 보는 서로 다른 두 구현이라
  신뢰도가 높다.
- 전체 이미지 기준 cross50 대비 체계차: V3 +0.490px, 팀원B M0 −0.774px, M2(바깥단) +0.277px,
  M5(계단맞춤) +0.368px — 모두 1px 미만으로 "같은 자릿수"(0.2절 "측정 위치는 오차원이
  아니다"와 같은 맥락, 엣지 *정의* 차이가 측정 *위치* 차이보다 작지는 않지만 비슷한 규모).

## 팀원B 결과와 대조
| 항목 | 일치 여부 |
|---|---|
| 81조합 문턱 sweep, 정답 봐도 최선 r=0.59(p=0.19) | **방향 일치**: 우리 9단계 sweep도 r=−0.417\~+0.448, 전부 유의하지 않음 |
| M0(50%경계)가 LOO 최선, 평균 찍기 못 이김 | **일치**: 우리 cross50·esf1·moment 모두 평균 찍기(5.78µm)보다 한참 나쁨(미리보기 수준) |
| NO4에서 V3가 B단 잡아 +5.7px | **일치**: 우리 grad_peak(+5.6px)·팀원A 엔진 재실행(+5.6px) 둘 다 재현 |
| M2\~M5(바깥단 계열)가 M0보다 나쁨 | **방향 일치**: 우리 esf2/esf3_outer(바깥단 계열)도 cross50보다 체계차·불안정성 모두 큼 |

## Codex adversarial-review 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] 재장착 변동을 측정하지 않은 상태에서 teacher 후보를 사실상 확정적으로 제시함 | **맞음** — 안정성 지표는 전부 같은 장착·광학계의 10장 반복 프레임만으로 쟀고, 독립 재장착 hold-out이 없다 | **반영함** | "한 줄 결론"과 "한계" 1번 항목을 다시 써서 결론 범위를 "현재 촬영 조건(13셋업, 재장착 반복 없음) 내 안정성 기준 후보"로 명시적으로 좁혔다. teacher **확정**이 아니라 **후보**라는 표현으로 통일했다. 독립 재장착 검증 자체는 다음 촬영(0.2절 "확정은 다음 촬영으로 한다")으로 미룬다 — 이번 실험 범위(10-08 데이터 하나)에서 새로 재촬영할 수는 없어서, 지적의 "재장착 반복을 추가"까지는 이번에 하지 못했고 결론 범위 축소로 반영했다. |
| [medium] `esf_fit`의 실패(`success=False`) fallback이 걸러지지 않고 그대로 유효값으로 집계됨 | **부분적으로 맞음** — 코드로 확인하니 `scan_definitions.py`가 `success`를 안 보고 항상 `centers`를 쓴 건 맞다. 다만 실제로 재보니 scipy `least_squares`의 `success`는 1890열 전부 `True`였다(실패율 0%) — "실패"가 전혀 없었다는 뜻이 아니라, **scipy가 "성공"이라고 보고하는 퇴화해(적합된 단 2개 이상이 거의 같은 위치로 겹침)가 실제 불안정의 원인**이었다 | **반영함(지적보다 더 깊이)** | `esf_fit`에 `MIN_STEP_SEP=1.5px` 단 붕괴 검사를 추가해 `success`를 scipy 수렴 여부 AND 단 간격 조건으로 재정의했다. `scan_definitions.py`는 `*_ok=False`를 NaN으로 기록(그대로 쓰지 않음)하고, `step1_summary.py`·`step2_threshold_sweep.py`는 NaN을 집계에서 뺀다. 그 결과 `esf2_outer`/`esf3_outer`의 진짜 실패율(67%/78%, 셋업별 0\~100%)이 드러났고 — 결과 CSV도 다시 생성해 모든 하류 수치(1·2·4·5단계)를 갱신했다. |
| [medium] `step=10` 교차검증이 cross50 한정이라 moment·ESF 순위를 보장 못 함 | **맞음** | **반영함** | 130장 전체를 `--step 2`로 재스캔(`scan_definitions.py --step 2`, `cache/e4/bearing_columns_step2.csv`)해 `step8_step_size_check.py`로 6개 정의 전부 비교했다. cross50·grad_peak·moment의 σ 차이는 0.0006\~0.0014px, **안정성 순위(esf1<cross50<moment<grad_peak<esf3_outer<esf2_outer)가 두 step 크기에서 완전히 같다**(`results/step_size_sensitivity.csv`). esf2_outer만 σ가 더 벌어지는데(0.158→0.228) 이미 "못 쓴다"로 결론 낸 정의라 전체 결론에는 영향 없다. |

동의하지 않는 지적 없음 — 3건 모두 코드·재실행으로 확인되는 실제 문제였다. 다만 1번(재장착
반복)은 "이번 실험 범위에서 바로 추가 검증할 수 있는 종류"가 아니라서 결론 범위를 좁히는
방식으로 반영했다 — 독립 재장착 hold-out 자체는 다음 촬영 때 가능하다.

## Codex `/codex:review` 1차 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] cross50·grad_peak가 E1의 탐색 창(±20px, `measure.py:126-127`)이 아니라 `WIN=45`를 써서, `edge_subpx`가 E1이 고르던 것과 다른 엣지(더 멀리 있는 반사 등)를 고를 수 있음 | **맞음** — 코드로 확인(`measure.py` `_measure_window`는 `edge_subpx(prof, tc-y0, 20, -1)`로 호출, `scan_definitions.py`는 전부 45였다) | **반영함** | `scan_definitions.py`에서 cross50·grad_peak(`edge_subpx` 호출)만 `EDGE_SUBPX_WIN=20`으로 분리했다(moment·ESF·문턱 sweep은 다단을 보려고 `MULTISTEP_WIN=45` 유지). 수정 후 베어링부 15열×130장 전부에서 `win=20`과 `win=45`의 cross50/grad_peak 값이 1e-6px 안으로 완전히 같음을 직접 재확인했다(실제 단 간격이 ±20px 안에 다 들어오기 때문으로 보인다) — 그래서 전체 파이프라인을 다시 돌려도 **모든 결과 수치가 이전과 동일했다**. 수치는 안 바뀌었지만 "다른 섹션·다른 촬영에서는 달라질 수 있는 구조적 위험"을 없앤 코드 수정이라 반영했다. |
| [P2] `step3_pin_f3_range.py`가 ESF1 실패 열의 NaN을 `median`/`max`/`min`에서 조용히 빼면서도 `f3_n`은 전체 이미지 수를 그대로 보고함 | **맞음** | **반영함** | `*_esf1_ok` 두 열을 명시적으로 걸러(`top_esf1_ok & bot_esf1_ok`) 유효 열이 없는 이미지는 통째로 뺀다. 결과 열을 `f3_n_total_images`/`f3_n_valid_images` 둘로 나눠 "몇 장이 실제로 쓰였는지"가 항상 드러나게 했다. 이번 데이터에서는 네 정의 전부 70/70(실패 없음)이라 수치 변화는 없다. |

동의하지 않는 지적 없음 — 둘 다 코드로 재현되는 실제 문제였다. 다만 둘 다 **이번 데이터에서는
수치 변화가 없었다**(직접 재실행으로 확인) — 구조적 위험을 없앤 방어적 수정이라는 점을
위 "근거" 칸에 정직하게 적었다.

## Codex `/codex:review` 2차 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P1] `esf_fit`이 `find_peaks_subpixel`로 방향성 극댓값을 충분히 못 찾아도(`n_steps=1`이면 0개 포함) `grad_peak` 위치로 때워 `least_squares`를 그대로 돌림 — `n_steps=1`일 때는 단 붕괴 검사(`min_gap`)가 비교할 상대 center가 없어 사실상 무의미해지고, scipy는 가짜 초기값에도 "수렴했다"고 보고하는 경우가 흔해서 `esf1_ok=True`로 잘못 기록될 수 있음 | **맞음** — 코드로 확인(`edge_methods.py:121-124`, `len(peaks) < n_steps`일 때 패딩 후 바로 fitting으로 진입, `n_steps=1`이면 `min_gap=inf`로 검사가 통과만 함) | **반영함** | `find_peaks_subpixel`이 찾은 극댓값이 `n_steps`개 미만이면(0개 포함) `least_squares`를 아예 돌리지 않고 즉시 `success=False`로 반환하도록 고쳤다. 전체 재스캔(step=10·step=2 둘 다) 후 확인: 베어링·핀의 `esf1` 실패율은 여전히 0%(이 섹션들은 항상 최소 1개의 방향성 극댓값이 있어서 — 이번 데이터에서는 가짜 성공이 실제로 생기지 않았다는 뜻)지만, **esf3_outer의 실패율은 78%→81%로 더 올라갔다**(가짜 초기값으로 어쩌다 "성공"하던 n=3 적합이 이제 정직하게 실패 처리됨) — esf3_outer의 잔차σ는 0.381→0.295px로 오히려 줄었다(나쁜 적합이 더 걸러져서). esf1/cross50/moment/grad_peak와 esf2_outer 수치, teacher 후보 결론은 변화 없다. |

동의하지 않는 지적 없음 — 코드로 재현되는 실제 문제였고, 지적대로 심각도가 높다(P1)고
본다 — esf1이 "실패율 0%로 가장 안정적"이라는 핵심 주장의 신뢰도에 직결되는 구멍이었다.
이번 데이터에서는 다행히 esf1 자체의 실패율이 바뀌지 않았지만, 그건 "운이 좋았다"는
뜻이지 수정이 불필요했다는 뜻은 아니다.

## 그림
- `figures/definitions_by_setup.png`: 정의별 셋업 중앙값 지름. grad_peak가 NO4에서만 튀고,
  esf2_outer·esf3_outer는 ESF 적합이 성립하는 셋업(대략 묶음1\~6)에서만 선이 그려진다 —
  나머지 셋업은 적합 자체가 실패해 값이 없다(단 붕괴 검사 반영 후).
- `figures/bottom_threshold_sweep.png`: 하단 문턱 10\~90% sweep, 셋업별 선 + 전체 median.

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음(규칙 8) — 이 실험의 가장 중요한 한계다.** 안정성(셋업 내 σ,
  열 잔차 σ)은 전부 "같은 장착·같은 광학계" 조건에서 찍은 10장 반복 프레임만으로 쟀다. 따라서
  지금 보이는 σ가 **엣지 정의 자체의 안정성인지, 아니면 이 촬영 1회에 공유된 정렬·ROI·초점
  상태의 반복성인지 이 데이터만으로는 분리할 수 없다**(Codex adversarial-review 반영 —
  한 줄 결론도 이 범위로 제한해 다시 썼다). 장착이 바뀌면 cross50/moment/esf1의 순위가
  뒤집힐 가능성을 배제할 수 없다. 독립 재장착·재촬영을 hold-out으로 추가하기 전까지는
  "teacher 확정"이 아니라 "현재 촬영 조건 내 최선 후보"로만 쓴다.
- **ESF 모델이 단마다 폭(sigma)을 공유하는 단순화를 썼다** — 실제로 단마다 블러 정도가
  다를 수 있어 esf2/esf3의 불안정성 일부는 이 단순화 탓일 수 있다(모델 자체의 한계인지,
  진짜 단이 없는 열에 억지로 맞춘 결과인지 이 실험만으로는 완전히 못 가른다).
- **ESF n단의 "바깥 단" 정의만 봤다** — 안쪽 단(가장 먼저 나타나는 단)을 쓰면 다른 안정성
  순위가 나올 수 있다. 팀원B M2\~M5와 비교하려고 바깥 단으로 통일했는데, teacher 후보로
  안쪽 단 ESF1(사실상 지금의 `esf1`과 같다)도 이미 봤으니 중복은 아니다.
- **`MIN_STEP_SEP=1.5px`(ESF n단 붕괴 판정 문턱)은 고정값으로 골랐다** — E3가 본 실제 단
  간격(5\~10px)보다 훨씬 보수적으로 잡아 "거의 같은 위치로 겹친 경우"만 잡아내려 했지만,
  이 값을 2\~3px로 올리면 esf2/esf3의 실패율이 더 올라갈 수 있다(민감도 분석은 안 함).
- **단 전환 비율은 이 구현으로는 답을 못 준다** — "단 후보"로 쓴 ESF3가 13셋업 중 8개에서
  적합 자체가 안 되니(위 2단계 참고), 애초에 비교할 셋업이 6개뿐이고 그 6개에서도 모두
  0.000이 나와 "셋업마다 얼마나 다른지" 보여주지 못했다. 더 나은 "단 후보" 정의(예: 셋업별로
  문턱을 조금씩 바꿔가며 직접 교차점을 추적)가 필요하다 — 이번 실험의 미해결 과제로 남긴다.
- **베어링부 스캔이 기본 `step=10`** — `step8_step_size_check.py`로 6개 정의 전부 `step=2`와
  대조해 순위가 보존됨을 확인했다(위 "방법론 메모"). 다만 이것도 **같은 130장**(재장착 반복
  없음)에 대한 교차검증이라, 위 첫 번째 한계와 독립이 아니다.
- 팀원A V2 엔진(`measurement_engine.py`)은 참조 목록에 있었지만 **이번엔 실행하지
  않았다** — 모듈을 import하면 import 시점에 `OUTPUT_DIR.mkdir(...)`가 무조건 실행돼
  Heeyeon 폴더 밖에 디렉터리를 만든다(규칙 0.5.1 위반 소지)와, 최상단에서 `ultralytics`를
  무조건 import해 우리 venv에 없는 의존성이 필요하다. V3만으로도 "팀원A 엔진 대조"라는
  목적은 충분히 달성했다고 보고, V2는 README에 사유만 남긴다(이 판단에 대한 재검토 요청을
  "Codex 리뷰에서 특히 봐 줬으면 하는 점"에도 남겼다).

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate   # 없으면: python3.11 -m venv .venv && pip install -r requirements.txt
cd E4_edge_benchmark
python3 scan_definitions.py            # --step 10(기본). cache/e4/{bearing,pin}_columns.csv (커밋 안 함)
python3 scan_definitions.py --step 2   # 한계 교차검증용. cache/e4/bearing_columns_step2.csv (커밋 안 함)
python3 step1_summary.py          # results/edge_method_benchmark.csv, results/esf_fit_failure_rate.csv
python3 step2_threshold_sweep.py  # results/bottom_threshold_sweep.csv, bottom_step_transition_by_setup.csv
python3 step3_pin_f3_range.py     # results/pin_f3_range.csv
python3 step4_correlation.py      # results/spearman_by_definition.csv (참고, teacher 선택 안 씀)
python3 step5_loo_mae.py          # results/loo_mae_by_definition.csv (참고, E6 전 미리보기)
python3 step6_reference_compare.py  # results/reference_compare.csv (팀원A V3 + 팀원B M0~M5)
python3 step7_figures.py          # figures/*.png
python3 step8_step_size_check.py  # results/step_size_sensitivity.csv (step=10 vs step=2, 6개 정의 전부)

# E1 영향 재확인(edge_methods.py·paths.py 추가가 기존 함수에 영향 없는지)
cd ../E1_module && python3 reproduce.py
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, matplotlib 3.11.2,
  **scipy 1.17.1(신규, ESF 최소제곱 적합에 필요 — 설치 전 알림, 0.5.7)**
- torch: 사용하지 않음
- 랜덤 시드: 순열검정 전부 `seed=20261013`(팀원B와 같은 날짜 기준 시드, `step2_threshold_sweep.py`
  `SEED`, `step4_correlation.py` `SEED`/`SEED+1`)

## 변경 파일 목록
```
M experiments/2nd/shaftlab/paths.py            (REFERENCE_BOX 상수 추가, 순수 추가)
A experiments/2nd/shaftlab/edge_methods.py     (level_crossing/moment_centroid/esf_fit, E4 전용)
M experiments/2nd/requirements.txt             (scipy==1.17.1 추가)
A experiments/2nd/E4_edge_benchmark/README.md
A experiments/2nd/E4_edge_benchmark/scan_definitions.py
A experiments/2nd/E4_edge_benchmark/step1_summary.py
A experiments/2nd/E4_edge_benchmark/step2_threshold_sweep.py
A experiments/2nd/E4_edge_benchmark/step3_pin_f3_range.py
A experiments/2nd/E4_edge_benchmark/step4_correlation.py
A experiments/2nd/E4_edge_benchmark/step5_loo_mae.py
A experiments/2nd/E4_edge_benchmark/step6_reference_compare.py
A experiments/2nd/E4_edge_benchmark/step7_figures.py
A experiments/2nd/E4_edge_benchmark/step8_step_size_check.py         (Codex adversarial-review 반영, 신규)
A experiments/2nd/E4_edge_benchmark/results/edge_method_benchmark.csv
A experiments/2nd/E4_edge_benchmark/results/esf_fit_failure_rate.csv  (Codex adversarial-review 반영, 신규)
A experiments/2nd/E4_edge_benchmark/results/bottom_threshold_sweep.csv
A experiments/2nd/E4_edge_benchmark/results/bottom_step_transition_by_setup.csv
A experiments/2nd/E4_edge_benchmark/results/pin_f3_range.csv
A experiments/2nd/E4_edge_benchmark/results/spearman_by_definition.csv
A experiments/2nd/E4_edge_benchmark/results/loo_mae_by_definition.csv
A experiments/2nd/E4_edge_benchmark/results/reference_compare.csv
A experiments/2nd/E4_edge_benchmark/results/step_size_sensitivity.csv  (Codex adversarial-review 반영, 신규)
A experiments/2nd/E4_edge_benchmark/figures/definitions_by_setup.png
A experiments/2nd/E4_edge_benchmark/figures/bottom_threshold_sweep.png
```
(`experiments/2nd/cache/e4/bearing_columns.csv`·`bearing_columns_step2.csv`·`pin_columns.csv`는
`.gitignore`의 `**/cache/`에 걸려 스테이징 대상 아님 — `scan_definitions.py`(`--step 2` 포함)로
재생성 가능, 각 1\~4MB. 팀원A V3 엔진의 오버레이 산출물(`cache/e4/teamA_v3_out/`, \~650MB)은
스칼라만 뽑은 뒤 스크립트가 자동으로 지운다.) 5MB 초과 파일 없음(가장 큰 산출물은
`reference_compare.csv` 32KB, 그림 두 장은 각 120KB 이하).

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- "한 줄 결론"을 "현재 촬영 조건 내 후보"로 범위를 좁힌 게 충분한 반영인지, 아니면 독립
  재장착 검증 없이는 아예 teacher "후보"라는 말조차 꺼내면 안 된다고 봐야 하는지 봐 달라.
- `MIN_STEP_SEP=1.5px`(ESF n단 붕괴 판정 문턱)를 고정값으로 고른 게 적절한지, 이 값에
  결과(esf2/esf3 실패율 67%/78%)가 얼마나 민감한지 — 민감도 분석은 이번에 안 했다.
- ESF n단 위치를 "바깥 단"(outer)으로 통일한 선택이 팀원B 대조에는 맞지만, teacher
  후보 비교(안쪽 단과의 차이)에 숨은 편향을 만들지 봐 달라.
- 팀원A V2 엔진을 import 안전성 문제(모듈 최상위의 `OUTPUT_DIR.mkdir`, `ultralytics`
  의존성)로 건너뛴 판단이 "엔진을 import해서 돌리되 수정하지 않는다"는 지시를 과하게
  좁게 해석한 건 아닌지 봐 달라.
