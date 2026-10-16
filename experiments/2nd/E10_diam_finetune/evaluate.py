"""E10 평가 집계: `finetune.py`가 낸 `diam_loso.csv`/`ng_eval.csv`/`edge_shift_raw.csv`를
모아 `results/summary.csv`(표 첫 줄 평균 찍기, 둘째 줄 teacher, 지시서 완료 기준)와
`results/edge_shift.csv`(변형별 열 이동 요약), `results/interpretation.csv`(7단계 해석
규칙, 결과를 보기 전에 고정한 세 조건)를 만든다.

**해석 규칙(지시서 E10 7단계, 결과를 본 뒤 바꾸지 않는다):** 아래 세 조건을 모두 만족하면
"실측 레이블이 정보를 담고 있을 가능성. 다음 촬영에서 재확인", 아니면 "이 데이터에서는
실측 레이블 학습이 셋업 잡음을 넘지 못함".
  1. 평균 찍기 대비 MAE가 3.4µm 이상 낮다(E6 동일 장착 반복성 바닥).
  2. 10 fold 중 7 fold 이상에서 평균 찍기보다 낮다.
  3. 열별 엣지 이동(상·하단 둘 다) 중앙값이 1px 이하다.
"""
import argparse
import os

import numpy as np
import pandas as pd

import data as D

WITHIN_MOUNT_REPEATABILITY_UM = 3.4260630898546243  # E6_calibration/results/within_mount_repeatability.csv, cross50


def load_results():
    diam = pd.read_csv(os.path.join(D.RESULTS_DIR, "diam_loso.csv"), encoding="utf-8-sig")
    ng = pd.read_csv(os.path.join(D.RESULTS_DIR, "ng_eval.csv"), encoding="utf-8-sig")
    edge_shift_raw = pd.read_csv(os.path.join(D.RESULTS_DIR, "edge_shift_raw.csv"), encoding="utf-8-sig")
    return diam, ng, edge_shift_raw


def baseline_mae(diam: pd.DataFrame) -> float:
    """'0_평균찍기'는 px를 전혀 안 보므로(계산 자체가 모델에 안 의존) 모든 변형에서
    같은 값이어야 한다 — 다르면 버그다(sanity check, assert). 전체 평균(집계) MAE —
    해석 규칙 1번(3.4µm 이상 개선)과 summary.csv 첫 행에 쓴다."""
    base = diam[(diam["eval_path"] == "model_px+calib") & (diam["method"] == "0_평균찍기")]
    per_variant_mae = base.groupby("variant")["abs_err_um"].mean()
    spread = per_variant_mae.max() - per_variant_mae.min()
    assert spread < 1e-6, f"평균찍기가 변형마다 다르다(버그 의심): {per_variant_mae.to_dict()}"
    return float(per_variant_mae.iloc[0])


def baseline_by_fold(diam: pd.DataFrame) -> pd.Series:
    """fold별 '0_평균찍기' abs_err_um(Codex adversarial-review [P2] 반영, 2026-10-16
    4차) — baseline 오차는 fold마다 많이 다르다(0.44~12.67µm, held-out 시편이 바뀌면
    학습 9시편의 평균도 바뀌므로). `fold_win_count`를 전체 평균(baseline_mae)과 비교하면
    "그 fold에서 평균 찍기를 이겼는가"를 answer하지 못한다 — fold별로 짝을 맞춰 비교해야
    한다. '0_평균찍기'는 모델에 안 의존하므로(위 baseline_mae) 아무 변형의 행을 써도
    같다 — 전부 모아 fold로 묶어 median을 쓴다(동일 값이라 mean과 같다)."""
    base = diam[(diam["eval_path"] == "model_px+calib") & (diam["method"] == "0_평균찍기")]
    return base.groupby("fold")["abs_err_um"].median()


def summarize_variant(diam: pd.DataFrame, variant: str, baseline: float, baseline_fold: pd.Series) -> dict:
    sub_direct = diam[(diam["variant"] == variant) & (diam["eval_path"] == "model_direct")].copy()
    mae_direct = float(sub_direct["abs_err_um"].mean())
    max_direct = float(sub_direct["abs_err_um"].max())
    # fold별 비교(Codex adversarial-review [P2] 반영, 2026-10-16 4차) — 전체 평균 baseline이
    # 아니라 **그 fold 자신의** 평균찍기 오차와 짝지어 비교한다.
    sub_direct["baseline_abs_err_um"] = sub_direct["fold"].map(baseline_fold)
    assert sub_direct["baseline_abs_err_um"].notna().all(), f"{variant}: fold가 baseline_by_fold에 없다"
    wins = sub_direct["abs_err_um"] < sub_direct["baseline_abs_err_um"]
    win_rate = float(wins.mean())
    win_count = int(wins.sum())

    row = dict(variant=variant, n_fold=len(sub_direct),
               MAE_um_model_direct=mae_direct, MAX_um_model_direct=max_direct,
               평균찍기대비_um=mae_direct - baseline, fold_win_count=win_count, fold_win_rate=win_rate)

    for method in ("0_평균찍기", "2_전체선형회귀", "3_물리스케일고정", "4_핀비율정규화선형"):
        sub_c = diam[(diam["variant"] == variant) & (diam["eval_path"] == "model_px+calib") &
                     (diam["method"] == method)]
        if len(sub_c):
            row[f"MAE_um_{method}"] = float(sub_c["abs_err_um"].mean())
    return row


def ng_summary(ng: pd.DataFrame, variant: str) -> dict:
    sub = ng[(ng["variant"] == variant) & (ng["eval_path"] == "model_direct")]
    out = {}
    for _, r in sub.iterrows():
        out[f"{r['specimen']}_판정"] = r["판정"]
        out[f"{r['specimen']}_방향"] = r["방향"]
        out[f"{r['specimen']}_err_um"] = round(r["err_um"], 1)
    return out


def edge_shift_summary(edge_shift_raw: pd.DataFrame) -> pd.DataFrame:
    """변형별 median(top_median)/median(bot_median) — 해석 규칙 3번이 쓰는 "열별 엣지
    이동 중앙값"은 fold별 중앙값들의 중앙값(대표값)이다. p95는 fold별 p95의 최댓값
    (가장 나쁜 fold)을 함께 보여 꼬리도 참고하게 한다."""
    g = edge_shift_raw.groupby("variant")
    out = g.agg(top_shift_median_px=("top_median", "median"), top_shift_worstfold_p95_px=("top_p95", "max"),
                bot_shift_median_px=("bot_median", "median"), bot_shift_worstfold_p95_px=("bot_p95", "max"))
    return out.reset_index()


def apply_interpretation(summary_row: dict, edge_shift_row: pd.Series) -> dict:
    cond1 = summary_row["평균찍기대비_um"] <= -WITHIN_MOUNT_REPEATABILITY_UM
    cond2 = summary_row["fold_win_count"] >= 7
    cond3 = (edge_shift_row["top_shift_median_px"] <= 1.0) and (edge_shift_row["bot_shift_median_px"] <= 1.0)
    verdict = ("실측 레이블이 정보를 담고 있을 가능성. 다음 촬영에서 재확인"
               if (cond1 and cond2 and cond3) else
               "이 데이터에서는 실측 레이블 학습이 셋업 잡음을 넘지 못함")
    return dict(cond1_mae_beats_repeatability=cond1, cond2_win_7of10=cond2,
                cond3_edge_shift_le1px=cond3, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=None,
                     help="finetune.py가 쓴 결과 CSV 위치이자 출력 위치(기본: E10_diam_finetune/results)")
    args = ap.parse_args()
    if args.results_dir:
        D.set_run_dirs(results_dir=args.results_dir)

    diam, ng, edge_shift_raw = load_results()
    baseline = baseline_mae(diam)
    baseline_fold = baseline_by_fold(diam)
    edge_shift = edge_shift_summary(edge_shift_raw)
    edge_shift_path = os.path.join(D.RESULTS_DIR, "edge_shift.csv")
    edge_shift.to_csv(edge_shift_path, index=False, encoding="utf-8-sig")
    print(f"저장: {edge_shift_path}")

    variants = [v for v in diam["variant"].unique() if not v.startswith("TEACHER_")]
    teacher_variants = [v for v in diam["variant"].unique() if v.startswith("TEACHER_")]
    ordered = teacher_variants + variants  # teacher를 표 위쪽에 두되, baseline은 별도 첫 행

    summary_rows = [dict(variant="0_평균찍기(baseline)", n_fold=10, MAE_um_model_direct=baseline,
                          MAX_um_model_direct=np.nan, 평균찍기대비_um=0.0, fold_win_count=np.nan,
                          fold_win_rate=np.nan)]
    interp_rows = []
    for variant in ordered:
        row = summarize_variant(diam, variant, baseline, baseline_fold)
        row.update(ng_summary(ng, variant))
        summary_rows.append(row)
        if not variant.startswith("TEACHER_"):
            es_row = edge_shift[edge_shift["variant"] == variant]
            if len(es_row):
                interp = apply_interpretation(row, es_row.iloc[0])
                interp_rows.append(dict(variant=variant, **interp))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(D.RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"저장: {summary_path} ({len(summary_df)}행)")

    interp_df = pd.DataFrame(interp_rows)
    interp_path = os.path.join(D.RESULTS_DIR, "interpretation.csv")
    interp_df.to_csv(interp_path, index=False, encoding="utf-8-sig")
    print(f"저장: {interp_path}")

    pd.set_option("display.width", 220)
    print()
    print("=== summary.csv (model_direct 기준 핵심 열) ===")
    print(summary_df[["variant", "MAE_um_model_direct", "평균찍기대비_um", "fold_win_count"]]
          .round(2).to_string(index=False))
    print()
    print("=== interpretation.csv ===")
    print(interp_df.to_string(index=False))


if __name__ == "__main__":
    main()
