"""E2 완료 기준 표: 증강×강도별 |Δ엣지|(상단·하단, cross50·grad_peak) 중앙값·최대값과
"측정 학습 허용"/"ROI·게이트 전용" 판정(|Δ| ≤ 0.03px 문턱, 엣지별로 따로).

**판정 축이 증강 종류에 따라 다르다:**
  - blur/brightness/contrast/gamma: Δ상단·Δ하단(위치가 실제로 얼마나 움직였는지)으로
    판정한다 — 이게 "라벨이 흔들리는 정도"다.
  - **회전은 Δ외경(지름)으로 판정한다.** 회전은 열 하나(고정 X)에서 상·하단 y좌표
    자체가 기하학적으로 크게 움직이는 게 정상이다(최대 1°에서 ~3px) — 그건 결함이 아니라
    "물체가 기울어졌으니 그 높이에서 본 모습이 달라진 것"뿐이다. 지시서가 "회전 후 기대
    외경은 직선적합 보정 기준으로 변하지 않아야 한다"고 못 박은 이유다. Δ상단·Δ하단은
    표에는 같이 적되 판정에는 안 쓴다.
  - **서브픽셀 평행이동은 판정이 아니라 양성 대조군 검증이다** — Δ상단≈Δ하단≈이동량(dy),
    Δ외경≈0이 나와야 파이프라인이 제대로 작동한다는 뜻이다. 0.03px 문턱을 그대로 적용하면
    "이동량만큼 정확히 움직인 정상 동작"이 전부 "게이트 전용"으로 잘못 찍히므로 따로 둔다.

**130장 전체 기준이고, "전체 130장 중앙값" 옆에 "셋업(묶음)별 중앙값 중 최악"도 같이
낸다**(Codex adversarial-review 반영) — 전체 중앙값만 보면 특정 셋업(초점·조도 조건)
에서만 크게 나오는 경우가 가려질 수 있어서다. `worst_setup_*`가 13개 셋업 각각의
중앙값 중 가장 큰 값이고, `worst_setup_묶음`이 그 셋업 번호다.

입력: results/aug_effect.csv
산출물: results/completion_criteria.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.03  # px
ROTATION_AUGS = {"rotation_deg"}
SHIFT_AUGS = {"subpixel_shift_y_px"}


def judge(med_abs: float) -> str:
    return "측정 학습 허용" if med_abs <= THRESH else "ROI·게이트 전용"


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "aug_effect.csv"), encoding="utf-8-sig")

    COLS = [("d_top_cross50_px", "top_cross50"), ("d_bot_cross50_px", "bot_cross50"),
            ("d_top_grad_peak_px", "top_grad_peak"), ("d_bot_grad_peak_px", "bot_grad_peak"),
            ("d_diam_cross50_px", "diam_cross50"), ("d_diam_grad_peak_px", "diam_grad_peak")]

    rows = []
    for (aug, intensity), g in df.groupby(["aug", "intensity"]):
        row = dict(aug=aug, intensity=intensity, n_images=len(g))
        by_bundle = g.groupby("묶음")
        for col, label in COLS:
            med = float(g[col].abs().median())
            mx = float(g[col].abs().max())
            row[f"{label}_med"] = med
            row[f"{label}_max"] = mx
            bundle_meds = by_bundle[col].apply(lambda s: s.abs().median())
            row[f"worst_setup_{label}_med"] = float(bundle_meds.max())
            row[f"worst_setup_{label}_묶음"] = int(bundle_meds.idxmax())
        row["bot_step_transition_frac_med"] = float(g["bot_step_transition_frac"].median())
        row["bot_step_transition_frac_max"] = float(g["bot_step_transition_frac"].max())

        if aug in ROTATION_AUGS:
            row["판정_축"] = "Δ외경(diam)"
            row["판정_cross50"] = judge(row["diam_cross50_med"])
            row["판정_grad_peak"] = judge(row["diam_grad_peak_med"])
            row["판정_cross50_셋업최악"] = judge(row["worst_setup_diam_cross50_med"])
        elif aug in SHIFT_AUGS:
            row["판정_축"] = "양성대조군(판정 아님)"
            row["판정_cross50"] = "검증용"
            row["판정_grad_peak"] = "검증용"
        else:
            row["판정_축"] = "Δ상단/Δ하단(엣지별)"
            row["판정_top_cross50"] = judge(row["top_cross50_med"])
            row["판정_bot_cross50"] = judge(row["bot_cross50_med"])
            row["판정_top_grad_peak"] = judge(row["top_grad_peak_med"])
            row["판정_bot_grad_peak"] = judge(row["bot_grad_peak_med"])
            # 전체 130장 중앙값은 통과해도, 특정 셋업(묶음)만 떼어 보면 그 셋업의 중앙값이
            # 문턱을 넘을 수 있다 — "셋업별 worst-case"를 따로 판정한다(Codex
            # adversarial-review 반영).
            row["판정_top_cross50_셋업최악"] = judge(row["worst_setup_top_cross50_med"])
            row["판정_bot_cross50_셋업최악"] = judge(row["worst_setup_bot_cross50_med"])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["aug", "intensity"])
    out_path = os.path.join(HERE, "results", "completion_criteria.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(out)}행)\n")

    pd.set_option("display.width", 240)
    print("=== blur/brightness/contrast/gamma: Δ상단·Δ하단(cross50) 기준 판정 (전체 130장 vs 셋업최악) ===")
    photometric = out[~out["aug"].isin(ROTATION_AUGS | SHIFT_AUGS)]
    print(photometric[["aug", "intensity", "top_cross50_med", "worst_setup_top_cross50_med",
                        "bot_cross50_med", "worst_setup_bot_cross50_med",
                        "판정_top_cross50", "판정_top_cross50_셋업최악",
                        "판정_bot_cross50", "판정_bot_cross50_셋업최악"]].round(4).to_string(index=False))

    flipped = photometric[(photometric["판정_top_cross50"] != photometric["판정_top_cross50_셋업최악"]) |
                           (photometric["판정_bot_cross50"] != photometric["판정_bot_cross50_셋업최악"])]
    print(f"\n전체-중앙값 판정과 셋업최악 판정이 갈리는 (증강,강도): {len(flipped)}개"
          + ("" if flipped.empty else f" — {flipped[['aug','intensity']].to_string(index=False)}"))

    print("\n=== NO1(묶음1=P1/F1, 묶음13=P2/F3 재촬영)이 다른 셋업보다 유별나게 흔들리는지 ===")
    no1 = df[df["묶음"].isin([1, 13])]
    other = df[~df["묶음"].isin([1, 13])]
    for col, label in [("d_top_cross50_px", "Δ상단"), ("d_bot_cross50_px", "Δ하단")]:
        print(f"  {label} |Δ| 중앙값 — 묶음1: {no1[no1['묶음']==1][col].abs().median():.4f}px, "
              f"묶음13: {no1[no1['묶음']==13][col].abs().median():.4f}px, "
              f"나머지 11개 셋업 평균: {other.groupby('묶음')[col].apply(lambda s: s.abs().median()).mean():.4f}px")

    print("\n=== 회전: Δ외경 기준 판정(Δ상단·하단은 참고만) ===")
    rot = out[out["aug"].isin(ROTATION_AUGS)]
    print(rot[["intensity", "diam_cross50_med", "diam_grad_peak_med", "top_cross50_med", "bot_cross50_med",
               "판정_cross50", "판정_grad_peak"]].round(4).to_string(index=False))

    print("\n=== 서브픽셀 평행이동(양성 대조군): Δ상단≈Δ하단≈이동량, Δ외경≈0 이어야 함 ===")
    shift = out[out["aug"].isin(SHIFT_AUGS)]
    print(shift[["intensity", "top_cross50_med", "bot_cross50_med", "diam_cross50_med"]].round(4).to_string(index=False))

    print("\n=== 하단 단 전환 비율이 높은 조건(0.1 이상) ===")
    trans = out[out["bot_step_transition_frac_med"] >= 0.1]
    print(trans[["aug", "intensity", "bot_step_transition_frac_med", "bot_step_transition_frac_max"]]
          .round(3).to_string(index=False) if len(trans) else "없음")


if __name__ == "__main__":
    main()
