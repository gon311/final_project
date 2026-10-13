"""E1 4단계(선택): 팀원B full_rows.csv(50% 경계)의 샤프트px·핀px를 파일명으로 맞춰
우리 cross50(d_line) 결과와 이미지별로 대조한다. 측정 열 범위(x0/x1)와 핀 열이 달라
차이가 나도 실패가 아니다 — 체계 차이(중앙값)와 셋업별 차이만 본다.

사용: python3 compare_teamB.py
산출물: results/vs_teamB.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ours = pd.read_csv(os.path.join(HERE, "results", "reproduce_diff.csv"), encoding="utf-8-sig")
    ours = ours[["file", "new_B_d_line"]].rename(columns={"new_B_d_line": "ours_B_d_line"})

    teamb_path = os.path.join(paths.MINWOOK_ROOT, "1008_field", "_out", "full_rows.csv")
    teamb = pd.read_csv(teamb_path, encoding="utf-8-sig")
    teamb = teamb.rename(columns={"파일": "file", "샤프트px": "teamB_shaft_px", "핀px": "teamB_pin_px",
                                   "묶음": "group", "역할": "role"})
    teamb = teamb[["file", "group", "role", "teamB_shaft_px", "teamB_pin_px"]]

    m = ours.merge(teamb, on="file", how="inner")
    m["diff_shaft_px"] = m["ours_B_d_line"] - m["teamB_shaft_px"]

    out_path = os.path.join(HERE, "results", "vs_teamB.csv")
    m.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"매칭된 이미지 {len(m)}장 (전체 ours={len(ours)}, teamB={len(teamb)})")
    print("샤프트px 차이(ours cross50 d_line - teamB 샤프트px) 전체 요약:")
    print(m["diff_shaft_px"].describe())
    print("\n묶음별 중앙값:")
    print(m.groupby("group")["diff_shaft_px"].median())


if __name__ == "__main__":
    main()
