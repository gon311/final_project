"""E5 6단계: 셋업 모니터 판정 (2신호).
  (a) 핀 px(2단계에서 고른 pin_relative 위치): F1/F2 뭉치와 F3 뭉치를 가르는 임계값
      (`focus_state`), 직전 묶음과 상태가 다른지(`focus_changed`), 묶음 안 프레임 순서
      기울기(초점 흐름, 팀원B 핀 -0.58px/묶음 기준).
      **주의: 핀 px 임계값은 F1 vs F3, F2 vs F3는 가르지만 F1 vs F2는 못 가른다**
      (2단계 bundle 평균: F1 268.95~269.69px, F2 269.44~269.51px로 겹친다). 그래서
      `focus_state`는 "F1F2"/"F3" 두 값뿐이고, 실제 F1->F2 전환(묶음4->5)은 이 신호로
      잡히지 않는다 — 분류표의 "초점" 열이 있어야만 안다.
  (b) 위치 신호: 팀원B 고정물(받침대 경계 xL·xR·간격, 창 아래 yB. `1011_camstate`) 기준으로
      묶음 경계(이전 묶음 마지막 장 -> 이 묶음 첫 장)에서 1px 넘는 이동이 있는지 본다.
두 신호(+ focus_changed)를 합쳐 "이 묶음은 재촬영 필요" 규칙 초안을 쓰고 10-08 데이터에
적용해 본다. Codex 2차 adversarial-review 반영: 기존에는 `f3_threshold_over`를 계산만
하고 `needs_reshoot`에 넣지 않았다 — `focus_changed`로 바꿔 넣는다.

입력: cache/pin_bearing_scalars.csv, results/pin_position_compare.csv, results/camera_shift_table.csv
산출물: results/setup_monitor.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache")

PIN_PX_F3_THRESHOLD = 270.2  # (a) 2단계 bundle 평균 F1/F2 최대 269.64, F3 최소 270.62의 중간값 근방
DRIFT_SLOPE_WARN_PX = 0.2    # (a) 묶음 안 10장 전체 변화량(기울기x9) 경고 임계값. NO7 +0.69px 참고.
MOUNT_JUMP_WARN_PX = 1.0     # (b) 팀원B 고정물 위치 임계값(팀원B 제안과 동일, state.csv는 sd~1px로 이 문턱에 맞는 정밀도)


def main():
    scal = pd.read_csv(os.path.join(CACHE_DIR, "pin_bearing_scalars.csv"), encoding="utf-8-sig")
    pos = pd.read_csv(os.path.join(HERE, "results", "pin_position_compare.csv"), encoding="utf-8-sig")
    df = scal.merge(pos[["file", "method_c_pin_relative", "window_clipped"]], on="file").sort_values(["묶음", "시각"])

    cam = pd.read_csv(os.path.join(HERE, "results", "camera_shift_table.csv"), encoding="utf-8-sig", index_col=0)

    rows = []
    prev_focus_state = None
    for g, d in df.groupby("묶음"):
        d = d.sort_values("시각")
        # window_clipped=True(pin_relative 창이 xR에 잘림)인 프레임은 핀 px 평균·흐름 계산에서
        # 뺀다 — 창이 좁아 지름이 덜 안정적일 수 있어서다(Codex 3차 adversarial-review 반영).
        d_clean = d[~d["window_clipped"]]
        n_clipped_excluded = len(d) - len(d_clean)
        if len(d_clean) < 3:  # 너무 많이 빠지면 통계가 의미 없으니 전체로 되돌린다
            d_clean = d
            n_clipped_excluded = 0
        pin_mean = d_clean["method_c_pin_relative"].mean()

        # (a) 묶음 안 프레임 순서 기울기(10장 전체 변화량 = 기울기 x 9, 팀원B reanalyze.py와 같은 정의)
        # clipped 프레임을 뺀 "남은 프레임들"의 시간순으로 다시 번호를 매긴다.
        order = np.arange(len(d_clean))
        slope = np.polyfit(order, d_clean["method_c_pin_relative"], 1)[0] if len(d_clean) >= 2 else 0.0
        drift_total = slope * (len(d_clean) - 1)

        focus_state = "F3" if pin_mean >= PIN_PX_F3_THRESHOLD else "F1F2"
        focus_changed = (prev_focus_state is not None) and (focus_state != prev_focus_state)
        prev_focus_state = focus_state

        # (b) 팀원B 고정물(받침대 경계) 묶음 경계 이동량 — camera_shift_table.csv에서 가져온다
        mount_xL = cam.loc[g, "delta_teamB_mount_xL_prev_bundle"]
        mount_xR = cam.loc[g, "delta_teamB_mount_xR_prev_bundle"]
        mount_yB = cam.loc[g, "delta_teamB_mount_yB_prev_bundle"]
        ours_xL_jump = cam.loc[g, "delta_ours_xL_prev_bundle"]  # 참고용(4px 격자라 거칠다)

        drift_warn = abs(drift_total) >= DRIFT_SLOPE_WARN_PX
        mount_vals = [v for v in (mount_xL, mount_xR, mount_yB) if not pd.isna(v)]
        position_warn = any(abs(v) > MOUNT_JUMP_WARN_PX for v in mount_vals) if mount_vals else False

        reasons = []
        if focus_changed:
            reasons.append(f"초점 상태 전환(-> {focus_state})")
        if drift_warn:
            reasons.append(f"묶음 안 흐름 {drift_total:+.2f}px")
        if position_warn:
            reasons.append(f"고정물 이동(xL {mount_xL:+.1f}, xR {mount_xR:+.1f}, yB {mount_yB:+.1f})")

        rows.append(dict(
            묶음=int(g), 초점=d["초점"].iloc[0], 시편=d["시편"].iloc[0],
            pin_px_mean=pin_mean, n_window_clipped_excluded=n_clipped_excluded,
            focus_state=focus_state, focus_changed=focus_changed,
            drift_total_px=drift_total, drift_warn=drift_warn,
            mount_xL_jump_px=mount_xL, mount_xR_jump_px=mount_xR, mount_yB_jump_px=mount_yB,
            ours_pin_xL_jump_px=ours_xL_jump, position_warn=position_warn,
            needs_reshoot=bool(focus_changed or drift_warn or position_warn),
            reason="; ".join(reasons) if reasons else "",
        ))

    out = pd.DataFrame(rows)

    # 회귀 테스트: 묶음7(F2->F3, 재초점 직후)은 focus_changed=True, 묶음13(F3->F3, NO1 재촬영)은 False여야 한다.
    row7 = out.loc[out["묶음"] == 7, "focus_changed"].iloc[0]
    row13 = out.loc[out["묶음"] == 13, "focus_changed"].iloc[0]
    assert row7 == True, f"묶음7 focus_changed는 True여야 하는데 {row7}"  # noqa: E712
    assert row13 == False, f"묶음13 focus_changed는 False여야 하는데 {row13}"  # noqa: E712
    print(f"[검증] 묶음7 focus_changed={row7} (기대 True), 묶음13 focus_changed={row13} (기대 False) -> 통과")

    out_path = os.path.join(HERE, "results", "setup_monitor.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"\n저장: {out_path}\n")
    print(out.round(3).to_string(index=False))
    print(f"\n재촬영 필요 판정된 묶음: {out.loc[out['needs_reshoot'], '묶음'].tolist()}")
    print("(묶음1은 이전 묶음이 없어 focus_changed·위치 신호 판정 불가 — drift만으로 판정)")
    print("주의: 묶음4->5의 실제 F1->F2 전환은 focus_state로 잡히지 않는다(F1/F2 핀 px가 겹침).")


if __name__ == "__main__":
    main()
