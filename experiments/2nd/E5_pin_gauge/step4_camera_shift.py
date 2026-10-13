"""E5 4단계: 카메라 이동은 새로 추정하지 않고 대조한다.
팀원B `1011_camstate/_out/state.csv`(받침대 경계 xL·xR·간격, 창 아래 경계 yB — 핀과 무관한
고정물)를 그대로 읽어 **묶음 경계(이전 묶음 마지막 장 -> 이 묶음 첫 장)** 기준 이동량 표를
만들고, 우리가 잰 핀 xL·샤프트 위치(x_from, top_y, bot_y)의 같은 기준 변화를 나란히 둔다.
(이전 묶음의 "첫 장"과 비교하는 건 이전 묶음 안의 흐름/드리프트를 묶음 전환 이동으로
잘못 섞을 수 있어서 못 쓴다 — Codex 2차 adversarial-review 반영. 참고용으로 첫-첫 비교값도
`*_first_to_first` 열에 따로 남긴다.)

입력: cache/pin_bearing_scalars.csv, MINWOOK_ROOT/1011_camstate/_out/state.csv
산출물: results/camera_shift_table.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache")


def first_frame_per_bundle(df, time_col="시각"):
    """묶음별 시간순 첫 행. `groupby().first()`는 열마다 따로 첫 non-NA 값을 골라
    다른 프레임의 값을 한 행에 섞을 수 있어 쓰지 않는다(Codex 4차 review 반영) —
    `nth(0)`은 그룹 안에서의 위치(0번째 행 전체)를 그대로 골라 같은 프레임의 값만 쓴다."""
    return df.sort_values(time_col).groupby("묶음").nth(0).set_index("묶음")


def last_frame_per_bundle(df, time_col="시각"):
    """묶음별 시간순 마지막 행. 위와 같은 이유로 `nth(-1)`을 쓴다."""
    return df.sort_values(time_col).groupby("묶음").nth(-1).set_index("묶음")


def main():
    scal = pd.read_csv(os.path.join(CACHE_DIR, "pin_bearing_scalars.csv"), encoding="utf-8-sig")
    cols = ["xL", "shaft_x_from", "shaft_top_y", "shaft_bot_y"]
    ours_first = first_frame_per_bundle(scal)[cols].add_prefix("ours_")
    ours_last = last_frame_per_bundle(scal)[cols].add_prefix("ours_")

    state_path = os.path.join(paths.MINWOOK_ROOT, "1011_camstate", "_out", "state.csv")
    teamb = pd.read_csv(state_path, encoding="utf-8-sig").rename(columns={"파일": "file"})
    tb_cols = ["xL", "xR", "간격", "yB"]
    teamb_first = first_frame_per_bundle(teamb, time_col="file")[tb_cols].add_prefix("teamB_mount_")
    teamb_last = last_frame_per_bundle(teamb, time_col="file")[tb_cols].add_prefix("teamB_mount_")

    table_first = ours_first.join(teamb_first).sort_index()
    table_last = ours_last.join(teamb_last).sort_index()
    # table_first와 같은 행 순서(묶음 오름차순)로, "직전 행"이 바로 이전 묶음의 마지막 장이 되게 한다.
    prev_bundle_last = table_last.shift(1)

    # (올바른 기준) 이전 묶음 마지막 장 -> 이 묶음 첫 장
    delta = table_first - prev_bundle_last
    delta.columns = [f"delta_{c}_prev_bundle" for c in delta.columns]

    # (참고용, 예전 방식) 이전 묶음 첫 장 -> 이 묶음 첫 장
    delta_first_to_first = table_first - table_first.shift(1)
    delta_first_to_first.columns = [f"delta_{c}_prev_bundle_first_to_first" for c in delta_first_to_first.columns]

    out = table_first.join(delta).join(delta_first_to_first)

    # 팀원B 고정물(받침대 경계) 세로 이동이 작은데(<3px) 우리 샤프트 y가 크게(>5px) 바뀌면
    # 카메라 이동으로 설명되지 않는 잔여 이동 -> 시편 놓임(앉은 높이) 후보로 표시한다(E5 4단계 지시).
    cam_y_stable = out["delta_teamB_mount_yB_prev_bundle"].abs() < 3
    shaft_y_jumped = out[["delta_ours_shaft_top_y_prev_bundle", "delta_ours_shaft_bot_y_prev_bundle"]].abs().max(axis=1) > 5
    out["residual_note"] = "설명 가능(카메라 이동과 동행)"
    out.loc[cam_y_stable & shaft_y_jumped, "residual_note"] = "시편 놓임(앉은 높이) 후보"
    out.loc[out.index == out.index.min(), "residual_note"] = "기준(이전 묶음 없음)"

    out_path = os.path.join(HERE, "results", "camera_shift_table.csv")
    out.to_csv(out_path, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(out)}행)")
    pd.set_option("display.width", 220)
    print("\n이전 묶음 마지막 장 -> 이 묶음 첫 장 (올바른 기준):")
    print(delta.round(2).to_string())
    print("\n참고: 이전 묶음 첫 장 -> 이 묶음 첫 장 (첫-첫 비교, 진단용):")
    print(delta_first_to_first.round(2).to_string())
    print("\n잔여 이동 판정:")
    print(out[["residual_note"]].to_string())

    print("\nNO5->NO6(묶음5->6) 예시(0.4절 '세로 +4px' 언급):")
    print(delta.loc[[6]].round(2).to_string())
    print("팀원B table_vs_cam.txt 인용: NO6 왼쪽받침대 세로 +7.0px, 오른쪽받침대 세로 -0.3px (NO3 기준, 다시 계산하지 않음)")


if __name__ == "__main__":
    main()
