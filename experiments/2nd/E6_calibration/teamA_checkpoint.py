"""E6 재현 체크포인트(지시서 방식1 설명): 팀원A V3 엔진 경로(E4가 이미 돌려 둔
`teamA_v3_measured_px`, `../E4_edge_benchmark/results/reference_compare.csv`, 다시 계산
안 함)로 묶음4·5·6(NO4~6) 3점 선형 캘리브레이션을 맞추면 **기울기 약 2.57µm/px, NG1·NG2
20장(각 10장) 전부 OK**가 나와야 한다(팀원B 재현값, 0.2절). 이 스크립트는 그 체크포인트
자체를 검증한다 — 통과하면 나머지 E6 결과(우리 자체 엣지 정의로 맞춘 결과)를 믿을 근거가
커진다.

입력: experiments/2nd/E4_edge_benchmark/results/reference_compare.csv(커밋됨, E4 산출물)
산출물: 콘솔 출력만(이 체크포인트는 다시 계산 않는 참조용이라 CSV로 저장하지 않는다).
"""
import os

import numpy as np
import pandas as pd

from calibration import TOL_HI, TOL_LO

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "..", "E4_edge_benchmark", "results", "reference_compare.csv")


def main():
    d = pd.read_csv(REF_PATH, encoding="utf-8-sig")
    # E4 reference_compare.csv는 정상 시편만 있다(0.3절 시편 있는 사진 전체) — NG는 없으므로
    # 분류표에서 다시 읽어 NG1·NG2의 teamA_v3_measured_px를 E4 캐시가 아니라 분류표 join으로
    # 가져와야 하지만, E4가 130장(정상+NG 전부)을 다 돌렸으므로 d에 NG도 포함돼 있어야 한다.
    assert 11 in d["묶음"].astype(int).unique() and 12 in d["묶음"].astype(int).unique(), \
        "reference_compare.csv에 NG1/NG2(묶음 11·12)가 없다 — E4 step6을 다시 확인해야 함"

    train = d[d["묶음"].isin([4, 5, 6])].groupby("묶음").agg(
        실측mm=("실측mm", "first"), px=("teamA_v3_measured_px", "median"))
    a, b = np.polyfit(train["px"].to_numpy(), train["실측mm"].to_numpy(), 1)
    slope_um_per_px = a * 1000
    print(f"팀원A V3 엔진 px, 묶음4·5·6 3점 선형: 기울기={slope_um_per_px:.3f} µm/px, "
          f"절편={b:.4f} mm  (팀원B 재현값: 약 2.57 µm/px)")
    print(train.round(3).to_string())

    ng = d[d["묶음"].isin([11, 12])].copy()
    ng["pred_mm"] = a * ng["teamA_v3_measured_px"] + b
    ng["판정"] = np.where((ng["pred_mm"] >= TOL_LO) & (ng["pred_mm"] <= TOL_HI), "OK", "NG")
    n_ok = int((ng["판정"] == "OK").sum())
    print(f"\nNG1·NG2 {len(ng)}장 중 OK 판정: {n_ok}장  (팀원B 재현값: 20/20 OK)")
    print(ng.groupby("묶음")["판정"].value_counts().to_string())


if __name__ == "__main__":
    main()
