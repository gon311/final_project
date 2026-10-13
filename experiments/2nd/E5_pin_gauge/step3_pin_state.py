"""E5 3단계: 핀이 언제 움직였는지 가린다.
분류표는 "묶음 1 안에서 이동"(P1)으로, 팀원B는 현장 LOG 70("1번 시편 후 핀 다시 놓음")과
위상상관(`1011_pincol/_out/shift.txt`)을 근거로 "묶음 1과 2 사이"로 본다.

우리가 보는 것: 묶음 1·2 각 10장을 촬영 순서(`시각`)대로 놓고 xL, pin_len(xR-xL 대신,
xR 자체는 우리 쪽에서 불안정해 길이로 본다), 핀 지름(E5 2단계에서 고른 pin_relative
방식)이 묶음 "안에서" 바뀌는지, 묶음 "사이에서" 바뀌는지를 본다. 카메라 이동 성분은
새로 추정하지 않고 팀원B 수치(오른쪽V블록 위상상관, 핀과 무관한 고정물)를 인용해 뺀다.

입력: cache/pin_bearing_scalars.csv, results/pin_position_compare.csv
산출물: results/pin_state.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache")

# 팀원B 1011_pincol/_out/shift.txt, 오른쪽V블록 dx (핀과 무관한 고정물, NO3 첫 장 기준 위상상관).
# 인용만 하고 다시 계산하지 않음(0.4절).
TEAMB_RIGHT_VBLOCK_DX = {1: -49.3, 2: -36.9}
TEAMB_PIN_AREA_DX = {1: -50.5, 2: -38.1}


def main():
    scal = pd.read_csv(os.path.join(CACHE_DIR, "pin_bearing_scalars.csv"), encoding="utf-8-sig")
    pos = pd.read_csv(os.path.join(HERE, "results", "pin_position_compare.csv"), encoding="utf-8-sig")
    scal = scal.merge(pos[["file", "method_c_pin_relative", "window_clipped"]], on="file")

    scal["pin_state"] = np.where(scal["묶음"] <= 1, "before_move", "after_move")

    for g in (1, 2):
        d = scal[scal["묶음"] == g].sort_values("시각")
        print(f"\n== 묶음{g} 촬영 순서(시각) ==")
        print(d[["file", "시각", "xL", "xR", "pin_len", "method_c_pin_relative", "window_clipped"]].to_string(index=False))
        n_clipped = int(d["window_clipped"].sum())
        if n_clipped:
            print(f"(참고: window_clipped=True {n_clipped}장 — pin_relative 창이 xR에 잘림, "
                  f"xL 분석에는 영향 없음(xL은 핀 지름과 별개로 검출됨))")
        print(f"묶음{g} 안 xL 범위: {d['xL'].min()}~{d['xL'].max()} (폭 {d['xL'].max() - d['xL'].min()}px) "
              f"-> {'안정' if d['xL'].max() - d['xL'].min() <= 6 else '불안정'}")

    b1_xL, b2_xL = scal.loc[scal["묶음"] == 1, "xL"].median(), scal.loc[scal["묶음"] == 2, "xL"].median()
    raw_delta = b2_xL - b1_xL
    cam_delta_pin_area = TEAMB_PIN_AREA_DX[2] - TEAMB_PIN_AREA_DX[1]
    cam_delta_right_vblock = TEAMB_RIGHT_VBLOCK_DX[2] - TEAMB_RIGHT_VBLOCK_DX[1]
    print(f"\n묶음1->묶음2 raw xL 변화: {raw_delta:+.0f}px")
    print(f"팀원B 카메라 이동 성분(핀주변 위상상관, dx 차이): {cam_delta_pin_area:+.1f}px")
    print(f"팀원B 카메라 이동 성분(오른쪽V블록, 핀과 무관한 고정물): {cam_delta_right_vblock:+.1f}px")
    resid = raw_delta - cam_delta_right_vblock
    print(f"카메라 이동(오른쪽V블록 기준) 뺀 잔여 핀 이동: {resid:+.1f}px")

    b1 = scal[scal["묶음"] == 1].sort_values("시각")
    within_b1_spread = b1["xL"].max() - b1["xL"].min()
    print(f"\n묶음1 안 xL 흔들림: {within_b1_spread}px (10장 전체)")
    print("-> 묶음1 '안에서' 핀이 뚜렷이 움직인 흔적은 보이지 않는다." if within_b1_spread <= 6 else
          "-> 묶음1 '안에서' 핀이 움직인 흔적이 있다(분류표 P1과 맞을 수 있음).")
    print(f"-> 카메라 이동을 빼도 묶음1->2 사이에 잔여 핀 이동 {resid:+.1f}px가 남는다. "
          "팀원B의 '묶음 1과 2 사이' 해석과 맞는 방향이다." if abs(resid) > 2 else
          "-> 카메라 이동을 빼면 묶음1->2 사이 잔여 핀 이동이 거의 없다.")

    out_cols = ["file", "묶음", "시편", "역할", "초점", "핀배치", "시각", "xL", "xR", "pin_len",
                "method_c_pin_relative", "window_clipped", "pin_state"]
    out = scal[out_cols].sort_values("file")
    out_path = os.path.join(HERE, "results", "pin_state.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(out)}행)")
    print(out["pin_state"].value_counts())


if __name__ == "__main__":
    main()
