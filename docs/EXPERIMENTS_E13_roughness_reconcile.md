# E13 거칠기 분석 재현·조정 — Claude Code 작업 지시서

> 작성 2026-10-27 · 저장소 `/Volumes/Lexar/Final/Dagyeom/git_upload/final_project` · 브랜치 `feature/4th_expr` · 작업 폴더 `experiments/4th/E13_roughness_reconcile/`
> 규칙 문서 `docs/PREREG_E13.md`를 **먼저 끝까지 읽는다.** 판정 기준·지표·통계는 PREREG가 정한다. 이 문서는 구현 방법과 산출물을 정한다. 두 문서가 다르면 PREREG를 따르고, 차이를 보고한다.
> 배경: 선행개발자님 분석(하단 엣지 1장 MAD vs Rq, r = +0.50)과 E11-A 보조 결과(하단 시편 고유 진폭 vs Rq, ρ = −0.62)가 반대 부호다. E13은 그 차이를 설명하는 **재현·조정** 실험이며 E11 결론을 바꾸지 않는다.
> 작업 규칙·보고 형식은 `docs/EXPERIMENTS_20261008.md` 0.5절(규칙 1\~14)·0.6절·2절을 따른다. 특히 규칙 5(시드·버전 기록), 10(평균 찍기 기준선은 이번에 해당 없음), 13(상관에는 순열검정)을 지킨다. 브랜치·작업 위치만 위 값으로 바꿔 읽는다.
> **한 번에 단계 하나만 진행한다.** 끝나면 7절 형식으로 보고하고 멈춘다. 다음 단계는 사용자가 지시할 때 시작한다.
> **새 측정·재촬영 없음. 원본 사진도 디코딩하지 않는다.** E11 캐시와 10/21 CSV만 쓴다.

---

## 0. 경로와 환경

저장소 루트를 `REPO`라고 한다.

| 항목 | 경로 |
|---|---|
| 작업 폴더 | `REPO/experiments/4th/E13_roughness_reconcile/` (이하 `E13`): `code/`, `results/`, `figures/`, `README.md` |
| 규칙·지시서 | `REPO/docs/PREREG_E13.md`, `REPO/docs/EXPERIMENTS_E13_roughness_reconcile_for_ClaudeCode.md`(이 문서) |
| 열별 엣지(읽기 전용) | `REPO/experiments/3rd/roughness/cache/edges_long.csv.gz` (gitignore. 열: filename, specimen, 단계, mount, 시각, x, y_top, y_bot. 없으면 `experiments/3rd/roughness/code/load_edges.py`로 다시 만들되 E11 폴더 밖 `E13/cache/`에 쓰고 보고) |
| 거칠기 | `REPO/experiments/3rd/roughness/results/roughness_clean.csv` (rough_id, specimen_1021, Ra, Rq, Rz, Rz_Ra, in_analysis) |
| 메타 | `REPO/experiments/3rd/E6_E10_20261021/results/values_by_image_1021.csv` (bearing_x_from, bearing_x_to, 시각, 단계) |
| E11 비교값 | `REPO/experiments/3rd/roughness/results/E11A_by_specimen.csv`, `E11A_summary.csv` |
| E11 코드(재사용, 수정 금지) | `REPO/experiments/3rd/roughness/code/e11a_edge_texture.py`(`robust_fit`, split-half·고정 패턴 계산), `e11a_sensitivity.py` |

- **`experiments/1st~3rd`, E12 폴더, 캐시 원본은 수정하지 않는다.** 출력은 모두 `E13` 아래에 둔다. `experiments/1st/`의 미추적 폴더는 건드리지 않는다.
- 환경: 사용자 Mac(Apple Silicon), CPU만. **`/usr/bin/python3`**(3.9.6, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, matplotlib 3.9.4)를 쓴다. 다른 파이썬을 쓰면 버전을 README에 적는다. 시드 42.
- 경로는 `code/paths.py` 한 곳에 둔다(`paths.py` 위치 기준 상대경로). 결과 CSV에는 절대경로를 쓰지 않는다.
- E11 함수를 쓸 때는 `sys.path`에 E11 `code/`를 추가해 import한다. 복사·수정하지 않는다. import가 어려우면 같은 산술로 옮겨 쓰고 그 사실을 PREREG 부록에 "실행 중 결정"으로 적는다.
- 스케일 0.026092 mm/px(1px = 26.092µm). 좌표는 모두 EXIF 적용 실물 방향, 하단 = `y_bot`.

### 0.0 git 규칙

- 시작 전에 `git branch --show-current`가 `feature/4th_expr`인지 확인한다. 다르면 멈추고 보고한다. 새 브랜치·merge·rebase·force·push는 하지 않는다.
- **사용자가 지시하기 전에는 커밋하지 않는다.** 단계가 끝나면 새 파일을 `git add`까지만 하고 보고한다(Codex가 커밋 전 변경을 리뷰한다).
- E13-0 보고 때 `docs/PREREG_E13.md`와 이 지시서도 함께 `git add`한다. PREREG가 결과보다 먼저 저장소에 들어가야 하므로, 사용자가 원하면 E13-0 전에 `docs(E13): 거칠기 재현·조정 사전 등록` 커밋을 먼저 할 수 있다.
- 커밋 지시를 받으면 단계 하나당 커밋 하나, 메시지는 `exp(E13-<단계>): <한 줄 요약>`.
- 작업 트리에 이 작업과 무관한 변경이 있으면 건드리지 않고 함께 스테이징하지도 않는다. 그림은 png 1장당 1MB 이하. 보고 직전에 `git status`와 스테이징된 파일 크기를 확인한다.

---

## 1. 진행 순서

| 순서 | ID | 내용 | 규모 | 성격 |
|---|---|---|---|---|
| 1 | E13-0 | 데이터 준비, 창·이미지·격자 점검, E11 정합 점검 | 작음 | 점검 |
| 2 | E13-1 | 선행개발자님 수치 재현 | 작음 | 재현 |
| 3 | E13-2 | 한 장 산포 성분 분해 | 작음 | **핵심(서술)** |
| 4 | E13-3 | 조건별 상관표 | 작음 | 탐색 |
| 5 | E13-4 | E11-A 하단 재계산 | 작음 | 사후 민감도 |
| 6 | E13-R | 종합 보고서 | 작음 | – |

- E13-1을 E13-3보다 먼저 한다. E13-1이 대표 MAD 정의와 재현 행을 정한다(PREREG 4절). E13-3 결과를 보고 그 정의를 바꾸지 않는다.
- 모든 단계가 CPU로 몇 분 안에 끝나야 한다. 10분을 넘을 것 같으면 시작 전에 예상 시간을 보고한다.

---

## E13-0. 데이터 준비와 점검

파일: `code/paths.py`, `code/e13_common.py`(창·점 추출·추세 제거·지표 함수), `code/e13_0_checks.py`.

- **방법**
  1. `edges_long.csv.gz`에서 본시험 NO2\~NO12 110장만 남긴다. `roughness_clean.csv`, `values_by_image_1021.csv`를 붙인다.
  2. `e13_common.py`에 PREREG 2.3·3절을 그대로 구현한다.
     - 창: `window_H()` = [8090.67, 8224]. `window_C(spec)` = 1번 이미지 c에서 [c − 76.65, c + 114.98]. `window_Crev(spec)` = [c − 114.98, c + 76.65]. 경계 포함(≤, ≥).
     - 점 추출: `pts`(x_k = 8008 + k × 272/30, 창 안의 k, `np.interp`로 y 보간), `cols`(창 안의 열).
     - 추세 제거: `none`, `poly1`(최소제곱 직선), `poly2`(E11 `robust_fit`).
     - 지표: `MAD_med`, `MAD_mean`, `STD`(ddof = 1), `Range`. `CV_diam`.
  3. 점검 표 `results/E13_0_checks.csv`
     - (a) 시편별 1번 이미지 파일명·c가 PREREG 2.3절 표와 같은지
     - (b) 이미지별 창 H·C·C_rev 안의 열 수, 창 H의 pts 점 수(14), 창 C의 오른쪽 끝 x(잘린 정도)
     - (c) 이미지별 x 격자 홀짝(PREREG 표와 같은지), NO8 안의 같은 홀짝 쌍 수(21)
     - (d) **E11 정합 점검**(PREREG 5절): 윗엣지, 이미지별 캐시 전체 열, poly2로 R_p, F_p를 계산(쌍·열 맞춤 규칙은 E13-2와 같다)하고 E11 `S_within_med`, `S_cross_med`와 시편 간 Spearman, 비의 중앙값을 낸다. NO8은 E11에서 보간했으므로 비교에서 빼고, 나머지 10시편으로 낸다.
- **완료 기준:** (a)\~(c) 표가 PREREG와 같음(다르면 멈추고 보고), (d) 수치. **거칠기와 아직 비교하지 않는다.**

---

## E13-1. 선행개발자님 수치 재현

PREREG 4절. 파일: `code/e13_1_reproduce.py`.

- **방법**
  1. NO2·NO9·NO10의 1번 이미지, 하단, 창 H에서 변형 8가지({MAD_med, MAD_mean} × {pts, cols} × {none, poly1})의 MAD·STD·Range를 px로 계산한다.
  2. 선행개발자님 값(MAD 0.573·0.811·0.930, STD 0.899·1.241·1.236, Range 3.589·4.521·5.293)과의 상대오차를 낸다.
  3. 재현 판정(MAD ±25%, 순서 NO2 < NO9 < NO10)을 변형마다 적용한다. PREREG 4절 규칙으로 **대표 MAD 정의와 재현 행**을 정해 `results/E13_1_summary.csv`에 적는다. 이 값이 E13-2·3의 입력이다.
  4. 보조: 11시편 CV_diam(창 H, 10장) vs 선행개발자님 Raw CV 표. Spearman ρ(NO9 포함·제외), 크기 비 중앙값.
- **산출물**
  - `results/E13_1_variants.csv`(변형 × 시편: n_points, MAD_px, STD_px, Range_px, 각 상대오차, reproduced)
  - `results/E13_1_cv.csv`(시편, CV_diam_H, rawcv_heeyeon)
  - `results/E13_1_summary.csv`(재현 여부, 재현 변형 목록, 대표 MAD 정의, 재현 행 설정, CV 비교 ρ)
  - `figures/E13_1_no2_no9_no10.png`: 세 시편 1번 이미지의 창 H 하단 엣지(cols 회색 선 + pts 점, 중앙값 수평선), 패널 제목에 MAD·STD·Range(우리 값 / 선행개발자님 값). 선행개발자님 인포그래픽과 같은 구성을 **실제 데이터로** 그린다.
- **완료 기준:** 판정 표와 대표 정의 한 줄, 그림 1장. 거칠기와 비교하지 않는다(E13-3에서 한다).

---

## E13-2. 한 장 산포 성분 분해 (핵심)

PREREG 5절. 파일: `code/e13_2_decompose.py`.

- **방법**
  1. 설정 3가지: 주 (C, poly2), 보조 (C, poly1), (H, poly2). 하단, cols.
  2. 이미지마다 창 안 열의 잔차 r_i를 만든다(추세는 창 안 열로 적합).
  3. 시편마다 V1_p, R_p, F_p, S_p, N_p와 몫을 계산한다.
     - cov는 두 이미지 공통 절대 x 열에서, 각 잔차에서 그 공통 열 평균을 빼고 ddof = 1로 계산한다.
     - 홀짝이 다른 쌍, 공통 열 30개 미만 쌍은 뺀다. 시편별 within 쌍 수, 시편 쌍별 cross 쌍 수를 기록한다.
     - F_p = q마다 이미지 쌍 cov 평균 → q들의 중앙값. 짝이 될 수 있는 q 개수를 기록한다.
  4. 주성분 판정(몫 중앙값 ≥ 0.5)을 설정마다 적용한다.
  5. 연결(서술): 시편 간 Spearman과 양측 순열 p(10000회, 시드 42)
     - 1번 이미지 MAD(같은 설정, 대표 정의) vs √max(S,0), √F, √N
     - 각 성분 vs Rq, Ra, Rz, 촬영 순서
- **산출물**
  - `results/E13_2_by_specimen.csv`(설정 × 시편: n_cols, n_within_pairs, n_cross_q, V1, R, F, S, N, share_S, share_F, share_N, MAD_img1, Rq, Ra, Rz, order)
  - `results/E13_2_links.csv`(설정 × 짝: rho, p_two_sided)
  - `results/E13_2_summary.csv`(설정별 몫 중앙값, 주성분, ρ(√S⁺, Rq))
  - `figures/E13_2_shares.png`: 주 설정, 시편을 Rq 순서로 놓은 S·F·N 몫 누적 막대(S가 음수면 0 아래로 표시)
- **완료 기준:** 몫 표, 주성분 한 줄, 연결 표, 그림 1장.
- **주의:** 통계 단위는 시편(n = 11)이다. 이미지·열 단위 p값을 내지 않는다. V1·R·F를 모두 같은 열 맞춤·같은 추세 제거로 계산한다(E11 Codex 지적: within과 cross의 열 집합 불일치).

---

## E13-3. 조건별 상관표 (탐색)

PREREG 6절. 파일: `code/e13_3_corr.py`.

- **방법**
  1. PREREG 6절 표의 8개 행 × 지표(대표 MAD, STD, Range)를 시편별로 계산한다. "10장" 행은 이미지별 지표의 시편 내 중앙값이다. CV_diam은 창 H, C로 따로 낸다.
  2. 각 행·지표에 대해 Ra·Rq·Rz와 Spearman ρ, 양측 순열 p, Pearson r, 촬영 순서와의 Spearman ρ를 낸다.
  3. 안정군 목록을 사용자가 실행 전에 주면 재현 행·대표 행을 그 부분집합으로 한 번 더 낸다. 안 주면 "안정군 목록 없음"으로 기록하고 넘어간다.
  4. PREREG 7절 정의로 결론 문장 A/B/C를 고르고, 주성분 문장과 고정 문장을 붙인다.
- **산출물**
  - `results/E13_3_metrics_by_specimen.csv`
  - `results/E13_3_corr.csv`(row, metric, rough_var, n, spearman_rho, p_two_sided, pearson_r, rho_order)
  - `results/E13_3_summary.csv`(대표 수치, 재현 행 Pearson r(Rq), 결론 문장)
  - `figures/E13_3_rep_vs_Rq.png`: 2패널(재현 행 / 대표 행), x = Rq, y = MAD(µm), 점에 시편명, 제목에 ρ·p·r. 점 색 = 촬영 순서.
- **완료 기준:** 상관표, 결론 문장 한 줄(PREREG 문장 그대로), 그림 1장. "유의" 표현은 쓰지 않는다.

---

## E13-4. E11-A 하단 재계산 (사후 민감도)

PREREG 8절. 파일: `code/e13_4_e11a_bottom.py`.

- **방법**
  1. E11 `e11a_edge_texture.py`의 split-half·고정 패턴·판정 함수를 import해 하단 엣지, 창 C(주)·C_rev(보조), poly2로 실행한다. 창은 시편별 1번 이미지 c 기준, 10장 모두 같은 창을 쓴다.
  2. NO8은 E11 부록 규칙(창 안 교집합 x 범위의 가장 큰 시작점부터 2px 격자 보간)을 따른다.
  3. 거칠기 Rz, Rq. 끝 3열 제외 변형을 함께 낸다(창 양 끝 각 3열).
  4. E11 PREREG 3.6절 1·2단계와 결론 문장 번호를 적되 모든 행에 `post_hoc = True`.
- **산출물:** `results/E13_4_by_specimen.csv`, `results/E13_4_summary.csv`(창, trim, rough_var, n_has_specific, rho, p_one_sided, conclusion_case, post_hoc)
- **완료 기준:** 요약 표. 결론 문장에 쓰지 않는다.

---

## E13-R. 종합 보고서

- `E13/README.md`를 E11 README와 같은 구성으로 쓴다: 0 결론 → 데이터와 사전 고정 → 단계별 표 → 사후 관찰 → 한계 → 팀 공유용 요약 → 파일 → 재현 순서.
- 결론 맨 앞에 **PREREG 7절 결론 문장**을 그대로 쓰고, 주성분 문장과 "E11 결론은 바뀌지 않는다"를 붙인다.
- 한계는 PREREG 10절을 빠짐없이 쓴다.
- **팀 공유용 요약(5줄 이내, 쉬운 말):** 선행개발자님 결과가 재현됐는지, 한 장 산포가 무엇으로 이루어졌는지, 거칠기와의 관계를 어떻게 말할 수 있는지. 스케일(272px ≠ 10.2mm, 실제 약 7.1mm) 정정 한 줄을 넣는다. 비난조 없이 사실만 쓴다.
- 다음 제안은 결과에서 나온 것만 쓴다(예: 선행개발자님 엣지 검출 방식 공유, 안정군 기준 문서화, 다점 거칠기 측정).

---

## 7. 보고 형식 (단계마다)

1. 한 줄 결론(해당 단계의 PREREG 문장 형식)
2. 핵심 수치 표(5행 이내)
3. PREREG와 다르게 한 것(없으면 "없음"). 있으면 PREREG 부록에 "실행 중 결정" 또는 "사후 변경"으로 기록했는지
4. 그림 경로
5. 한계 한두 줄(PREREG 10절 중 해당하는 것)
6. 실행 명령(재현용)
7. 변경 파일 목록(`git status` 요약)과 5MB 넘는 파일 유무
8. **Codex 리뷰에서 특히 봐 줬으면 하는 점** 1\~3개(한 줄씩, `/codex:adversarial-review` focus text로 붙여 넣을 수 있게)

---

## 8. 리뷰가 지적하기 쉬운 부분 (미리 지킬 것)

- 통계 단위는 시편(n = 11)이다. 열·이미지 단위로 p값을 내지 않는다.
- 결과를 보고 창·추세 차수·MAD 정의·재현 허용폭(±25%)·주성분 기준(0.5)·"양의 상관 재현" 기준(r ≥ 0.30)을 바꾸지 않는다. 바꾸면 사후 변경으로 기록하고 원래 결과와 나란히 낸다.
- E13-2에서 V1·R·F는 같은 열 맞춤·같은 추세 제거로 계산한다. 홀짝이 다른 쌍을 보간으로 억지로 맞추지 않는다(보간은 평활 효과로 잡음 분산을 줄인다).
- 재현 행의 Pearson r은 선행개발자님 값과 비교하기 위한 참고값이다. 판정은 PREREG 7절 정의만 따른다.
- 안정군 기준을 추측해서 만들지 않는다.
- E13-4는 사후 민감도다. E11 결론을 바꾸는 문장을 쓰지 않는다.
- 하단 = `y_bot`(EXIF 적용 좌표). 원본 파일 좌표와 섞지 않는다.
