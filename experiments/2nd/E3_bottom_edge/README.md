# E3. 하단 다단 엣지 진단

## 목적
하단의 여러 단(A/B, 경우에 따라 C)이 언제·왜 갈리는지 조건별로 정리한다. GT가 없는 상태에서
"어느 단이 물리적 외곽에 가까운지"와 "다단의 원인"을 간접 증거로 추정한다. 결론은 모두
**가설**이다 — 확정은 다음 촬영(하단 가림·필름 유무를 번갈아 촬영, 기준 샤프트)으로 한다.

## 한 일
1. `shaftlab/multistep.py`(신규): `find_peaks_subpixel()`(gradient 극댓값 상위 n개,
   `edge_subpx`와 같은 포물선 보간), `edge_multistep_features()`(cross50·배경/안쪽 밝기·
   엣지 폭까지 묶음), `scan_section_multistep()`(열별 스캔). `measure_section`/`_measure_window`
   등 E1이 "수정하지 않는다"고 못 박은 함수는 건드리지 않고 별도 모듈로 뒀다.
2. `shaftlab/measure.py`에 `R_COARSE` 상수 추가(우측 직선부 탐색 구간, `eda_measure.py`의
   "우측 긴 축부(참조 구간)" 호출과 동일 — 순수 추가라 E1 재현에 영향 없음, 재확인함).
3. `scan_profiles.py`: 130장(0.3절 기준)에서 베어링부 상·하단, 핀 상·하단, 우측 직선부(R_)
   하단의 열별 다단 특징을 뽑는다. `results/bottom_peaks.csv`(지시서가 명시한 산출물,
   3.8MB)만 커밋하고, 나머지 대용량 원자료(top/핀/R_, 각 3.7\~6.4MB)는 `cache/e3/`에 둔다
   (0.6절: 10MB 미만이어도 "이미지×열" 단위 대용량은 요약만 커밋).
4. `step1_feature_summary.py` \~ `step4_error_correlation.py`: 아래 "결과 요약" 참고.

## 방법론 메모 (모든 단계에 공통)
`n_peaks_detected`(2단 vs 3단 개수)만으로는 변별력이 없다 — **핀(절벽 하나인 게 확실한
대상)도 열의 99%에서 "3개"가 잡힌다.** 블러·노이즈가 진짜 엣지 주변에 작은 흔들림을 만들고,
우리 극댓값 탐지기(`_local_maxima_indices`)는 그런 흔들림도 "극댓값"으로 센다. **실제로
다단인지 가르는 건 "가장 약한 극댓값이 가장 강한 것 대비 얼마나 센가"(weakest/strongest
세기 비)다** — 노이즈면 5% 안팎, 진짜 경쟁하는 단이면 15\~20%대로 뚜렷이 갈린다(아래 표).
E4에서 teacher 정의를 고를 때 이 구분을 참고할 만하다.

| 구간 | weakest/strongest 중앙값 | 해석 |
|---|---|---|
| 핀 상단 | 0.012 | 절벽 하나 |
| 베어링부 상단 | 0.024 | 절벽 하나 |
| 핀 하단 | 0.053 | 절벽 하나 |
| 우측 직선부(R_) 하단 | 0.113 | 약한 다단 |
| 베어링부 하단 | 0.189 | 뚜렷한 다단 |

**이 지표를 믿어도 되는지 독립적으로 검증했다(`validate_peak_detection.py`, Codex 4차
adversarial-review 반영):**
- **합성 단일-절벽 대조군:** 실제 이미지의 점대점(point-to-point) 잡음(1차 차분 기반
  추정 — 창 전체 `std()`는 느린 밝기 추세까지 섞어 fg σ≈5, bg σ≈13로 심하게 과대추정되니
  쓰지 않는다)으로 "진짜 절벽 하나" 프로파일을 500번 합성했다. **91.4%는 애초에 극댓값이
  3개도 안 나왔고**(평균 1.72개 — 절벽 하나로 올바르게 분류됨), 드물게(8.6%) 3개가 나와도
  weakest/strongest 중앙값은 0.005로 베어링부 하단(0.189)과 전혀 다르다. **실제 잡음
  수준에서는 노이즈만으로 weakest/strongest 비율이 베어링부 하단 수준까지 올라가지
  않는다** — 지표가 "진짜 노이즈"와 "진짜 다단"을 구분한다는 근거다. (처음 시도는 창 전체
  `std()`로 잡음을 과대추정해 합성 대조군이 오히려 더 다단처럼 보이는 실패를 겪었다 — 이
  실패와 원인 진단도 정직하게 남겨 둔다. 1차 차분으로 바로잡은 뒤에야 제대로 검증됐다.)
- **`cluster_radius`·`min_sep` 민감도:** 대표 5장에서 `cluster_radius`∈{15,22,30}px,
  `min_sep`∈{2,3,5}px **9가지 조합 전부**에서 "베어링부 하단 > 베어링부 상단, 핀 하단"
  순위가 그대로 유지됐다(`results/peak_detection_validation.csv`). 파라미터 선택이
  결론을 뒤집지 않는다.
- 다만 두 검증 모두 **이미지 1장(`20261008_151701.jpg`) 기준**이다 — 모든 130장에 대한
  전수 민감도 검증은 아니다(한계, 아래 참고).

## 결과 요약

### 1\~3단계 — 단 개수·위치, cross50이 고르는 단 (`results/bottom_features_by_setup.csv`)
- 단 간격(gap_12, gap_23)은 묶음 대부분에서 5\~10px, 팀원B가 본 단 간격(6\~13px, y≈4222/4229/4235)과
  **같은 자릿수로 일치한다.**
- **cross50이 고르는 단은 거의 모든 묶음에서 "2단"(위치 순서 가운데)이다** — 우리 정의(전역
  최강 gradient 기준 bg/fg로 cross50을 계산)에서는 셋업·초점에 관계없이 매우 일관됐다.
  이 집계에서 `cross50_step_ambiguous=True`(cross50이 선택된 극댓값 중 어느 것과도 2px
  넘게 떨어진, 묶음별 0.0\~2.0%)인 열은 뺐다 — 빼도 결론은 그대로다(Codex 4차
  adversarial-review 반영).
  **팀원B가 관찰한 "F1·F2의 묶음2\~5는 작은 단, 묶음1·6·8·9·10은 급경사에 걸림"이라는
  묶음별 전환은 우리 데이터·정의로는 재현되지 않는다.** 두 분석의 50% 교차 구현(배경·전경
  샘플링 창, 블러 커널)이 달라서일 수 있다 — 어느 쪽이 "맞는지"는 이 비교만으로 가릴 수 없다.
- 하단 흐림 폭(10\~90%) 중앙값은 묶음별로 11\~15px, 샤프트 안쪽 밝기 비율은 0.24\~0.41로
  0.2절의 알려진 값(0.3\~0.4)과 같은 범위다.
- **묶음별 weakest_strongest_ratio_med는 0.038\~0.598로 폭이 넓다**(방법론 메모의 0.189는
  전체 열을 합친 중앙값일 뿐, 묶음마다 "다단인 정도"가 상당히 다르다). 묶음8은 0.038로
  거의 절벽 하나에 가깝고, 묶음4는 0.598로 매우 뚜렷한 다단이다 — 이 편차 자체가 왜
  생기는지는 이번 분석으로 설명하지 못한다(가설 후보로만 남긴다).

### 4단계 — 핀 상·하단에도 같은 분석 (`figures/pin_vs_shaft_bottom_profile.png`)
**핀 상·하단 모두 weakest/strongest 비율이 낮다(0.01\~0.05, 베어링부 상단과 같은 수준)
— 팀원B의 "핀 위·아래는 1\~2px 절벽 하나"가 재현된다.** 다단은 핀에는 없고 샤프트
하단(그리고 R_, 아래 6단계)에만 있다. 핀은 깊이·화면 위치가 샤프트와 다르다는 점은
감안해야 하지만, "다단 자체가 핀에 없다"는 사실은 다단이 핀과 공유하지 않는 조건
(가림 여부, 0.2b의 "가리지 않은 백라이트" 가설)과 더 맞는 방향이다.

### 5단계 — 상·하단 대칭성, 가림 가설 (`figures/top_bottom_alignment.png`)
상단(가림막으로 가림)의 weakest/strongest 비율(0.024)이 핀(가림 여부와 무관하게 절벽
하나, 0.01\~0.05)과 같은 수준이고, 하단(가리지 않음)은 7.8배 더 높다(0.189). **그림으로도
뚜렷하다** — 상단을 뒤집어 정렬하면 급격한 한 번의 전이로 끝나는 반면, 하단은 전이가
0\~20px에 걸쳐 완만하게 꺾이며 두 단계로 진행한다. **"가리지 않은 백라이트가 샤프트
아래를 스친다"는 팀원B 추정과 맞는 방향이다** — 다만 이 비교는 "가림 여부와 다단이
같이 간다"는 상관일 뿐, 메커니즘(정말 백라이트가 스치는지)까지 확인한 건 아니다(가설).
(참고: 그림에서 상단 곡선에 x≈-10\~-5 부근 작은 굴곡이 보인다 — 0.2절이 언급한 상단의
"링잉(ISP 샤프닝 흔적)"과 같은 방향일 수 있으나 한 장만으로는 확정할 수 없다.)

### 6단계 — 우측 직선부(R_) 하단 (`results/r_bottom_features_by_setup.csv`)
R_ 하단의 weakest/strongest 비율(0.113)은 베어링부 하단(0.189)보다 약하지만 핀·상단
(0.01\~0.05)보다는 뚜렷이 높다 — **다단 구조가 R_에도 "약하게" 나타난다.** 베어링부와
직선부가 같은 조명 기하(가리지 않은 백라이트)를 공유한다는 점과 맞는 방향이지만, 베어링부
쪽이 왜 더 강한지(형상 차이? 거리 차이?)는 이 데이터로 알 수 없다(가설).

### 7단계 — 특징과 오차의 상관 (permutation test)
정상 10개 셋업(NO1\~NO10, 묶음1\~10; 묶음13=NO1 재촬영은 중복이라 제외)에서, cross50(B_d_line,
물리 스케일 0.0367mm/px 가정 후 평균 오프셋 제거 — **교정식 아님, 이 상관 검정 전용**)로
얻은 오차(µm)와 상관을 봤다.

| 특징 | r | 순열 p (n=10, 5000회) |
|---|---|---|
| 하단 흐림 폭(10\~90%) | -0.266 | 0.465 |
| 하단 흐림 폭(90->50%) | +0.149 | 0.688 |
| 샤프트 안쪽 밝기 비율 | **-0.685** | **0.028** |

- **팀원B의 r=-0.69(아래 흐림 폭, `1012_edgecheck`)는 우리 데이터·정의로 재현되지
  않는다**(r=-0.266, p=0.465, 유의하지 않음). 흐림 폭을 구하는 세부 방식(10\~90% 기준,
  bg/fg 샘플링 창)이 서로 다른 게 원인일 수 있다.
- 대신 **샤프트 안쪽 밝기 비율에서 비슷한 크기의 상관(r=-0.685)**을 찾았는데, 이건
  팀원B가 약한 후보로 본 변수다(|r|≤0.22, 자신들의 분석에서는). **이 결과를 과신하면
  안 된다** — n=10이고 특징을 3개 사전에 정하지 않고 사후에 비교했으니, 다중비교를
  보정하면(예: 3개 중 하나, Bonferroni 임계 0.05/3≈0.017) p=0.028은 **유의 기준을
  통과하지 못한다.** 이 결과는 가설로만 남기고, 다음 촬영에서 다시 확인할 후보로 기록한다
  (지시서 요구사항).

## 팀원B 결과와 대조
| 항목 | 일치 여부 |
|---|---|
| 단 간격 6\~13px(y≈4222/4229/4235) | **대체로 일치**(우리 gap 중앙값 5\~10px, 같은 자릿수) |
| cross50이 묶음별로 "작은 단"/"급경사"를 오간다 | **불일치**(우리는 거의 항상 "2단", 묶음 무관) |
| 핀 위·아래는 절벽 하나 | **일치**(weakest/strongest 비율로 재확인) |
| 아래 흐림 폭 vs 오차 r=-0.69 | **불일치**(우리 r=-0.266, p=0.465) — 대신 안쪽 밝기 비율에서 비슷한 크기 상관 발견(가설) |
| 안쪽 밝기, 위 흐림 폭 등은 약한 후보(|r|≤0.22) | **부분 불일치**: 안쪽 밝기는 우리 데이터에서 더 강했다(다중비교 미보정) |
환산 방식: 둘 다 cross50(50% 경계) 계열이지만 세부 구현(샘플링 창·블러)이 달라 직접
비교에는 체계 차이가 섞여 있을 수 있다(0.4절).

## Codex 4차 adversarial-review 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [high] 다단 검출이 진짜 단계가 아니라 극댓값 상위 3개를 강제로 채운 것(prominence/노이즈 기준 없음) | **맞음** | **반영함** | README "방법론 메모"에 `validate_peak_detection.py`의 독립 검증을 추가했다. 실제 점대점 잡음 수준(1차 차분 추정)으로 만든 합성 "진짜 절벽 하나"는 91.4%가 애초에 극댓값 3개를 못 채우고(평균 1.72개), 드물게 채워도 weakest/strongest=0.005로 베어링부 하단(0.164)과 확실히 다르다. `cluster_radius`·`min_sep` 9개 조합 전부에서 "하단>상단,핀" 순위도 유지된다. 다만 검증은 이미지 1장 기준이라 전수는 아니다(한계에 추가). |
| [high] `cross50_step`이 실제 전이 단이 아니라 "선택된 피크 중 가장 가까운 것"이라 교차가 피크 사이에 끼거나 진짜 전이 피크가 top-n에서 탈락해도 항상 1\~3단 라벨이 붙음 | **맞음** | **반영함** | `edge_multistep_features()`에 `cross50_step_ambiguous`를 추가했다(cross50이 가장 가까운 선택 피크에서도 2px 넘게 떨어지면 True — 실제 분포의 상위 1\~3%에 해당하는 데이터 기반 문턱). `step1_feature_summary.py`가 모호한 열을 "단 지배" 집계에서 빼도록 고쳤다(묶음별 0.0\~2.0%, 빼도 결론 불변). |
| [medium] 필수 요약 CSV(`bottom_features_by_setup.csv`)의 세기비 컬럼이 README의 weakest/strongest와 다른 계산(strongest/2nd)을 담고 있어 재현 불가 | **맞음**(코드로 확인: `strength_ratio_strongest_2nd`는 `max/두번째로 큰 값`인데 README는 `min/max`를 설명) | **반영함** | `step1_feature_summary.py`를 고쳐 두 지표를 **이름을 구분해서 둘 다** 저장한다 — `weakest_strongest_ratio_med`(README가 실제로 인용하는 지표, min/max)와 `top1_top2_strength_ratio_med`(기존 지표, max/2nd, 이름 명확화). 이제 커밋된 CSV만으로 README의 0.007\~0.589 범위가 재현된다. |

동의하지 않는 지적 없음 — 3건 모두 코드로 재현되는 실제 문제였다.

## Codex 5차 review(`/codex:review`) 반영 표
| 지적 | 사실 확인 | 반영 여부 | 근거 |
|---|---|---|---|
| [P2] `find_peaks_subpixel()`이 부호 조정된 gradient(`seg = g*sign`)가 0 이하인 극댓값도 후보로 받아들여, 이 엣지 방향과 반대로 기운 구간의 "그나마 덜 음수인 점"까지 단으로 셈. 커밋된 데이터에서 3단으로 잡힌 15,376열 중 1,884열(12.3%)이 이런 비양수 세기 "단"을 포함하고 있어 weakest/strongest 비율과 `n_peaks_detected`가 오염됨 | **맞음** — `(s <= 0).any(axis=1).sum()`으로 직접 재현. 또 이 버그가 이전 `validate_peak_detection.py`의 9-조합 민감도 표에서 `pin_bottom`(min_sep=5)에 `-0.010`이라는 물리적으로 말이 안 되는 음수 비율로 이미 겉으로 드러나 있었다 | **반영함** | `multistep.py`의 `find_peaks_subpixel()`에 `seg[k] > 0`인 극댓값만 후보로 삼는 필터를 추가. 전체 재스캔(130장) 후 4개 산출물 모두 비양수 세기 행 0건 확인(bottom 0/15438, top 0/15438, pin 0/21300, r_bottom 0/25870). step1\~4, `validate_peak_detection.py` 전부 재실행: 이전의 `-0.010` 아티팩트는 사라지고 `0.035/0.036/0.038`로 정상화됐다. 핵심 수치가 소폭 변했지만(베어링부 하단 weakest/strongest 0.164→0.189, R_ 0.106→0.113, 상단/핀은 거의 불변) 모든 정성적 결론("베어링부 하단만 진짜 다단")은 유지되거나 오히려 더 또렷해졌다(비율 7.2배→7.8배). E1 재현도 재확인: PASS(`max|diff|`는 1e-12px 수준의 부동소수점 잡음). |

동의하지 않는 지적 없음 — 이번 라운드도 1건뿐이었고, 코드로 직접 재현되는 실제 버그였다.
(수정 과정에서 자체적으로 발견한 별개 문제도 하나 고쳤다: `validate_peak_detection.py`의
마지막 설명 문장이 베어링부 하단 비율을 `0.164`로 하드코딩해 둬서 이번 재계산 값(`0.189`)과
어긋났다 — 계산된 변수(`r_bot`)를 직접 포맷하도록 고쳐 항상 최신 값을 출력하게 했다.)

## 그림
- `figures/pin_vs_shaft_bottom_profile.png`: 대표 열 1개에서 핀(절벽 하나) vs 샤프트
  하단(완만한 2단 전이) 밝기 프로파일 비교.
- `figures/top_bottom_alignment.png`: 같은 열의 상단(뒤집음)과 하단을 정렬 — 상단은
  급격한 전이 하나, 하단은 0\~20px에 걸친 두 단계 전이.

## 한계와 다음 촬영에서 확인할 점
- **13개 셋업, 재장착 반복 없음** — 최종 결론을 주장하지 않는다(규칙 8). 모든 결론은
  가설 표기(지시서 요구사항).
- 다단 검출(`find_peaks_subpixel`의 `cluster_radius=22px`)은 휴리스틱이다 — 계단 전체
  폭(20\~25px) 지식에 맞췄지만, 다른 조명·초점 조건에서는 다시 조정이 필요할 수 있다.
  `validate_peak_detection.py`로 민감도·합성 대조군 검증을 했지만 **둘 다 이미지 1장
  기준**이다 — 130장 전체, 13묶음 전체에 대한 전수 검증은 아니다.
- 7단계 상관은 n=10, 사후 비교 3개로 다중비교 미보정 — 가설 수준이다고 이미 README에
  명시했다. 교정식으로 쓰지 않는다(지시서 요구사항).
- cross50이 "항상 2단"이라는 우리 결과와 팀원B의 "묶음별로 바뀐다"는 결과가 다른 이유
  (구현 차이 vs 실제 다른 현상)는 이 데이터로 가릴 수 없다 — 다음 촬영에서 두 구현을
  나란히 돌려 보면 좋다.
- 묶음별 weakest_strongest_ratio_med가 0.007\~0.589로 크게 벌어지는 이유(왜 어떤 묶음은
  거의 절벽 하나처럼 보이는지)는 설명하지 못했다 — 가설 후보로만 남긴다.
- 확정은 다음 촬영(하단 가림·필름 유무를 번갈아 촬영, 기준 샤프트)으로 한다(지시서 요구사항).

## 실행 명령 (재현용)
```bash
cd experiments/2nd
source .venv/bin/activate   # 없으면: python3.11 -m venv .venv && pip install -r requirements.txt
cd E3_bottom_edge
python3 scan_profiles.py            # results/bottom_peaks.csv, cache/e3/{top,pin_edge,r_bottom}_peaks.csv
python3 step1_feature_summary.py    # results/bottom_features_by_setup.csv
python3 step2_pin_and_symmetry.py   # figures/pin_vs_shaft_bottom_profile.png, figures/top_bottom_alignment.png
python3 step3_r_section.py          # results/r_bottom_features_by_setup.csv
python3 step4_error_correlation.py  # 콘솔 출력만(교정식 아님, CSV화 안 함)
python3 validate_peak_detection.py  # results/peak_detection_validation.csv + 합성 대조군 콘솔 출력

# E1 영향 재확인(shaftlab에 R_COARSE·multistep.py 추가 후에도 PASS인지)
cd ../E1_module && python3 reproduce.py
```

### 환경·패키지 버전
- Python 3.11.9 (venv: `experiments/2nd/.venv`)
- opencv-python-headless 5.0.0.93, numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, matplotlib 3.11.2
- torch: 사용하지 않음
- 랜덤 시드: 7단계 순열검정 `seed=20261013`(`step4_error_correlation.py`의 `SEED`)

## 변경 파일 목록
```
M experiments/2nd/shaftlab/measure.py          (R_COARSE 상수 추가, 순수 추가라 기존 함수 불변)
A experiments/2nd/shaftlab/multistep.py        (다단 엣지 저수준 함수, E3 전용;
                                                 cross50_step_ambiguous 플래그 포함;
                                                 Codex 5차 review 반영 — seg[k]>0 필터 추가)
A experiments/2nd/E3_bottom_edge/README.md
A experiments/2nd/E3_bottom_edge/scan_profiles.py
A experiments/2nd/E3_bottom_edge/step1_feature_summary.py   (weakest/strongest 지표 이름 수정,
                                                              cross50_step_ambiguous 제외 집계)
A experiments/2nd/E3_bottom_edge/step2_pin_and_symmetry.py
A experiments/2nd/E3_bottom_edge/step3_r_section.py
A experiments/2nd/E3_bottom_edge/step4_error_correlation.py
A experiments/2nd/E3_bottom_edge/validate_peak_detection.py  (합성 대조군 + 민감도 검증, 신규)
A experiments/2nd/E3_bottom_edge/results/bottom_peaks.csv
A experiments/2nd/E3_bottom_edge/results/bottom_features_by_setup.csv
A experiments/2nd/E3_bottom_edge/results/r_bottom_features_by_setup.csv
A experiments/2nd/E3_bottom_edge/results/peak_detection_validation.csv
A experiments/2nd/E3_bottom_edge/figures/pin_vs_shaft_bottom_profile.png
A experiments/2nd/E3_bottom_edge/figures/top_bottom_alignment.png
```
(`experiments/2nd/cache/e3/{top,pin_edge,r_bottom}_peaks.csv`는 `.gitignore`의 `**/cache/`에
걸려 스테이징 대상 아님 — `scan_profiles.py`로 재생성 가능, 각 3.7\~6.4MB.) 5MB 초과 파일
없음(`bottom_peaks.csv`가 가장 커서 약 3.7MB, 그림 두 장은 각 100KB 이하).

## Codex 리뷰에서 특히 봐 줬으면 하는 점
- `validate_peak_detection.py`의 합성 대조군·민감도 검증이 **이미지 1장**만 기준으로
  한다 — 13묶음 전체·130장 전체로 넓혀도 "weakest/strongest가 노이즈와 다단을 가른다"는
  결론이 유지되는지 봐 달라.
- cross50_step_ambiguous 문턱(2px)을 실제 분포(상위 1\~3%)로 정했는데, 이 문턱 자체가
  1\~3단계의 "cross50이 항상 2단"이라는 결론에 숨은 편향을 만들 가능성이 있는지 봐 달라.
- 7단계 오차 정의(물리 스케일 가정 + 평균 오프셋 제거)가 "교정식이 아니다"라고 못박아
  뒀지만, 그래도 이 정의 자체가 특정 방향으로 상관을 왜곡할 구조적 이유가 있는지 봐 달라.
- 5·6단계(상·하단 대칭성, R_ 비교)가 "가림 가설과 맞는 방향"이라고 결론 냈는데, weakest/
  strongest 비율 하나로 "가림"과 "다른 원인"(예: 베어링부·직선부·핀의 형상 차이 자체)을
  구분할 수 없다는 한계가 충분히 명확히 적혀 있는지 봐 달라.
