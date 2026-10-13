"""E3 7단계: 셋업 중앙 특징(특히 하단 흐림 폭)과 실측 대비 오차의 상관을 계산해 팀원B의
r = -0.69(`1012_edgecheck`)가 재현되는지 본다. 규칙 13에 따라 순열 p값을 붙인다.
**이 상관은 보정식으로 쓰지 않고, 다음 촬영에서 다시 확인할 후보로만 기록한다.**

오차 정의: B_d_line(cross50, px) * 물리 스케일(0.0367mm/px) = 측정mm(물리 스케일 가정).
평균 찍기류 오프셋(정상 10개 평균의 치우침)을 뺀 뒤 실측mm과의 차이를 오차(µm)로 쓴다
(교정식이 아니라 이 상관 분석 전용 — E6에서 실제 교정 방식을 따로 비교한다).
정상 10개 = NO1~NO10(묶음1~10). 묶음13(NO1 재촬영)은 중복이라 제외(0.3절).

입력: paths.EDA_MEASURE_RAW_CSV(B_d_line), results/bottom_features_by_setup.csv
산출물: 없음(콘솔 출력만 — 수치가 작고 '보정식으로 쓰지 않는다'는 지시라 CSV화하지 않음)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSICAL_SCALE_MM_PER_PX = 0.0367
NORMAL_BUNDLES = list(range(1, 11))  # NO1~NO10 (묶음1~10), 묶음13(NO1 재촬영) 제외
SEED = 20261013
N_PERM = 5000


def main():
    raw = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig")
    raw = raw[raw["묶음"].notna()]
    raw["묶음"] = raw["묶음"].astype(float).astype(int)
    px_by_bundle = raw[raw["묶음"].isin(NORMAL_BUNDLES)].groupby("묶음").agg(
        B_d_line_med=("B_d_line", "median"), 실측mm=("실측mm", "first"))

    feat = pd.read_csv(os.path.join(HERE, "results", "bottom_features_by_setup.csv"), encoding="utf-8-sig")
    feat = feat.set_index("묶음")
    feat.index = feat.index.astype(int)

    df = px_by_bundle.join(feat[["edge_width_10_90_med", "edge_width_90_to_50_med", "inner_ratio_med"]],
                            how="inner")
    assert len(df) == 10, f"정상 10개가 아니라 {len(df)}개 — 묶음 매칭 확인 필요"

    df["measured_mm"] = df["B_d_line_med"] * PHYSICAL_SCALE_MM_PER_PX
    offset = (df["measured_mm"] - df["실측mm"]).mean()
    df["error_um"] = (df["measured_mm"] - offset - df["실측mm"]) * 1000

    print("정상 10개 셋업 (오차는 평균 오프셋 제거 후, 교정식 아님 — 상관 분석 전용):")
    print(df[["B_d_line_med", "실측mm", "error_um", "edge_width_10_90_med"]].round(3).to_string())

    def permutation_test(x, y, seed, n_perm):
        r_obs = float(np.corrcoef(x, y)[0, 1])
        rng = np.random.default_rng(seed)
        r_perm = np.empty(n_perm)
        y_arr = np.asarray(y, dtype=float)
        for i in range(n_perm):
            r_perm[i] = np.corrcoef(x, rng.permutation(y_arr))[0, 1]
        p = float((np.abs(r_perm) >= abs(r_obs)).mean())
        return r_obs, p

    for feat_col, label in [("edge_width_10_90_med", "하단 흐림 폭(10~90%)"),
                             ("edge_width_90_to_50_med", "하단 흐림 폭(90->50%)"),
                             ("inner_ratio_med", "샤프트 안쪽 밝기 비율")]:
        r_obs, p = permutation_test(df[feat_col], df["error_um"], SEED, N_PERM)
        print(f"\n{label} vs 오차: r={r_obs:+.3f}, 순열검정 p={p:.3f} (n=10, 셔플 {N_PERM}회, seed={SEED})")

    print("\n팀원B 인용(`1012_edgecheck/pattern.py`, 다시 계산하지 않음): 아래 흐림 폭만 오차와")
    print("r=-0.69 (보정하면 LOO 15.5 -> 10.8µm). 다른 후보(아래 단 차이, 안쪽 밝기, 위 흐림 폭,")
    print("앉은 높이, 배경 밝기)는 |r|<=0.22.")


if __name__ == "__main__":
    main()
