"""E5 0단계: 130장(분류표에 시편 정보가 있는 이미지, 0.3절) 전체를 한 번만 디코딩해
핀 전체 구간의 열별 cross50·grad_peak 엣지(`results/pin_profile.csv`, 1단계 산출물)와,
이후 단계(2~6)가 재사용할 스칼라 요약(`cache/pin_bearing_scalars.csv`, 커밋 안 함)을 만든다.

사용: (experiments/2nd/.venv 활성화 후) python3 scan_pins.py
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import measure_bearing, measure_pin, measure_pin_profile, paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

META_COLS = ["시각", "묶음", "시편", "역할", "등급", "실측mm", "초점", "핀배치"]


def main():
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]  # eda_pin_gauge.py와 같은 기준, 0.3절(메모 사진 제외)
    files = sorted(meta.index)

    profile_rows = []
    scalar_rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        m = meta.loc[fn]
        meta_dict = {c: m[c] for c in META_COLS}

        prof = measure_pin_profile(gray, margin_frac=0.0, step=2)
        # xL=x_from, xR=x_to(검출된 핀 구간의 원시 경계, coarse_section 결과 그대로) —
        # measure_pin()이 반환하는 xL/xR과 같은 정의다. x_per_col[-1](마지막으로 "측정한" 열)을
        # 쓰면 step=2 격자 때문에 항상 최대 1px(여기서는 짝수 폭이라 항상 2px) 작게 나와
        # measure_pin()의 클램프 기준(xR)과 어긋난다 — Codex 3차 adversarial-review 반영.
        xL = int(prof["x_from"])
        xR = int(prof["x_to"])
        s = prof["x_per_col"] - xL
        for x, ss, tc, bc, tp, bp in zip(
            prof["x_per_col"], s, prof["top_cross_per_col"], prof["bot_cross_per_col"],
            prof["top_peak_per_col"], prof["bot_peak_per_col"]
        ):
            profile_rows.append(dict(
                file=fn, **meta_dict, x=int(x), s=int(ss),
                top_cross=tc, bot_cross=bc, width_cross=bc - tc,
                top_peak=tp, bot_peak=bp, width_peak=bp - tp,
            ))

        bearing = measure_bearing(gray)
        # position_method="auto": step2에서 "현 EDA 자동 구간" 방식의 재현 검산용으로 쓴다.
        # measure_pin()의 기본값 자체는 이미 E5 2단계 결론(pin_relative)으로 바꿔 둔 상태라
        # 여기서 기본값에 맡기면 더는 "auto"가 아니다.
        pin_auto = measure_pin(gray, position_method="auto")

        # 교차 산출물 일치 검사(Codex 3차 adversarial-review 반영): measure_pin_profile()과
        # measure_pin()은 같은 coarse_section(PIN_COARSE)을 각자 다시 돌려 xL/xR을 구한다.
        # 같은 이미지에서 둘이 다르면 코드 어딘가 경계 정의가 또 어긋난 것이므로 바로 멈춘다.
        if pin_auto["xL"] != xL or pin_auto["xR"] != xR:
            raise AssertionError(
                f"{fn}: scan_pins의 xL/xR({xL},{xR})이 measure_pin()의 xL/xR"
                f"({pin_auto['xL']},{pin_auto['xR']})과 다름 — 경계 정의가 어긋났다"
            )

        scalar_rows.append(dict(
            file=fn, **meta_dict, xL=xL, xR=xR, pin_len=xR - xL,
            shaft_x_from=bearing["x_from"], shaft_x_to=bearing["x_to"],
            shaft_top_y=bearing["top_y"], shaft_bot_y=bearing["bot_y"], shaft_d_line=bearing["d_line"],
            pin_auto_d_line=pin_auto["d_line"], pin_auto_d_peak_median=pin_auto["d_peak_median"],
        ))
        print(k, fn, round(time.time() - t0, 2), "xL=", xL, "xR=", xR, "n_cols=", prof["n_cols"], flush=True)

    profile_df = pd.DataFrame(profile_rows)
    profile_path = os.path.join(HERE, "results", "pin_profile.csv")
    profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {profile_path} ({len(profile_df)}행, {len(files)}장)")

    scalar_df = pd.DataFrame(scalar_rows)
    scalar_path = os.path.join(CACHE_DIR, "pin_bearing_scalars.csv")
    scalar_df.to_csv(scalar_path, index=False, encoding="utf-8-sig")
    print(f"저장: {scalar_path} ({len(scalar_df)}행, cache — 커밋 안 함)")


if __name__ == "__main__":
    main()
