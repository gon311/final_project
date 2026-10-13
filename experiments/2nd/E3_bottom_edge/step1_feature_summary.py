"""E3 1~3단계: 하단 다단 엣지 특징을 셋업·초점별로 요약한다.
  1. 열마다 하단 gradient 극대 상위 3개의 위치·세기, 단 간격·세기 비, 셋업별 2단/3단 비율.
  2. 하단 흐림 폭(10~90%, 90->50%), 샤프트 안쪽 밝기(배경 대비), 배경 밝기, cross50이 걸린 단.
  3. 셋업·초점별 분포 비교. gradient-peak(=가장 강한 단)와 cross50이 어느 단을 고르는지 비율.
     팀원B 관찰(F1·F2의 NO2~5=묶음2~5는 작은 단, NO1·6·8·9·10=묶음1·6·8·9·10은 급경사에
     걸림, `1008_field` 계열 분석)을 재현하는지 확인한다.

**세기 비 열 이름 주의(Codex 4차 adversarial-review 반영):** 이전 버전은
`strength_ratio_med`라는 이름으로 "가장 강한 극댓값 / 두 번째로 강한 극댓값"을 저장해
놓고 README는 "가장 약한 극댓값 / 가장 강한 극댓값"(weakest/strongest, `step2`·`step3`가
쓰는 지표)을 설명해서, 커밋된 CSV로 README의 핵심 수치를 재현할 수 없었다. 이번에 두
지표를 **이름을 구분해서 둘 다** 넣는다 — `top1_top2_strength_ratio_med`(기존 지표,
강한 두 극댓값의 비)와 `weakest_strongest_ratio_med`(step2·3과 같은 정의, README가
실제로 인용하는 지표).

**cross50_step 모호성(Codex 4차 adversarial-review 반영):** cross50이 선택된 극댓값
중 어느 것과도 가깝지 않으면(`cross50_step_ambiguous`, `multistep.py` 참고) 그 열의
"몇 단인가" 라벨을 신뢰할 수 없다 — "단 지배" 집계에서 제외하고 비율만 따로 적는다.

입력: results/bottom_peaks.csv (scan_profiles.py 산출물)
산출물: results/bottom_features_by_setup.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

TEAMB_SMALL_STEP_BUNDLES = {2, 3, 4, 5}
TEAMB_STEEP_STEP_BUNDLES = {1, 6, 8, 9, 10}


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "bottom_peaks.csv"), encoding="utf-8-sig")

    # 단 간격·세기 비 (3단이 잡힌 열만)
    df3 = df[df["n_peaks"] == 3].copy()
    strengths = df3[["peak1_strength", "peak2_strength", "peak3_strength"]]
    df3["gap_12"] = df3["peak2_y"] - df3["peak1_y"]
    df3["gap_23"] = df3["peak3_y"] - df3["peak2_y"]
    df3["top1_top2_strength_ratio"] = strengths.max(axis=1) / strengths.apply(lambda r: sorted(r)[-2], axis=1)
    df3["weakest_strongest_ratio"] = strengths.min(axis=1) / strengths.max(axis=1)

    # cross50_step_ambiguous=True인 열은 "몇 단에 걸리는가" 집계에서 뺀다
    df_unambig = df[~df["cross50_step_ambiguous"]]

    agg = df.groupby("묶음").agg(
        초점=("초점", "first"), 시편=("시편", "first"), n_cols=("x", "count"),
        frac_2steps=("n_peaks", lambda s: float((s == 2).mean())),
        frac_3steps=("n_peaks", lambda s: float((s == 3).mean())),
        cross50_step_ambiguous_frac=("cross50_step_ambiguous", "mean"),
        edge_width_10_90_med=("edge_width_10_90", "median"),
        edge_width_90_to_50_med=("edge_width_90_to_50", "median"),
        inner_ratio_med=("inner_ratio", "median"),
        bg_level_med=("bg_level", "median"),
    )
    step_frac = df_unambig.groupby("묶음")["cross50_step"].value_counts(normalize=True).unstack(fill_value=0.0)
    step_frac.columns = [f"cross50_step{int(c)}_frac" for c in step_frac.columns]
    agg = agg.join(step_frac)

    gaps = df3.groupby("묶음").agg(gap_12_med=("gap_12", "median"), gap_23_med=("gap_23", "median"),
                                    top1_top2_strength_ratio_med=("top1_top2_strength_ratio", "median"),
                                    weakest_strongest_ratio_med=("weakest_strongest_ratio", "median"))
    agg = agg.join(gaps)

    out_path = os.path.join(HERE, "results", "bottom_features_by_setup.csv")
    agg.to_csv(out_path, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"저장: {out_path}\n")
    print(agg.round(3).to_string())

    step_cols = [c for c in agg.columns if c.startswith("cross50_step") and c.endswith("_frac")
                 and c != "cross50_step_ambiguous_frac"]
    print(f"\ncross50_step_ambiguous 비율: 묶음별 {agg['cross50_step_ambiguous_frac'].min():.3f}"
          f"~{agg['cross50_step_ambiguous_frac'].max():.3f} (모호한 열은 아래 '단 지배' 집계에서 뺐다)")

    print("\n단 지배(cross50이 어느 단에 걸리는지, 묶음별 최빈값, 모호한 열 제외):")
    dominant = agg[step_cols].idxmax(axis=1)
    for g, d in dominant.items():
        step_num = d.replace("cross50_step", "").replace("_frac", "")
        marker = ""
        if g in TEAMB_SMALL_STEP_BUNDLES:
            marker = " <- 팀원B: 작은 단 예상"
        elif g in TEAMB_STEEP_STEP_BUNDLES:
            marker = " <- 팀원B: 급경사 예상"
        print(f"  묶음{int(g)}: 주로 {step_num}단{marker}")

    print("\n(참고) 팀원B 관찰은 '작은 단 vs 급경사'라는 정성적 구분이고, 우리는 '어느 단이 cross50")
    print("기준 전역 최강 gradient인가'(위치 순서 기준)로만 본다 — 정확히 같은 분류는 아니다.")
    print("\nweakest_strongest_ratio_med(README가 '다단인지' 판단에 쓰는 지표) 범위:",
          f"{agg['weakest_strongest_ratio_med'].min():.3f}~{agg['weakest_strongest_ratio_med'].max():.3f}")


if __name__ == "__main__":
    main()
