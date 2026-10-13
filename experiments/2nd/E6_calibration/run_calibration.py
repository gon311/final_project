"""E6 메인: 방식(0~4) x 정의(cross50/grad_peak/moment) x 사진 선택(all10/last5/no7excl) x
평가 분할(A 시편 LOO/B 팀원A NO4~6->NO7~10)을 모두 돌려 `results/calibration_loso.csv`에
낸다. 방식 0(평균 찍기)은 모든 조합에서 같이 계산해 비교 기준으로 쓴다(규칙 10).

방식 1(3점 선형)은 항상 묶음4·5·6만 학습한다(팀원A 방식 그대로) — "LOO"라는 말이 이
방식에는 어색하지만(학습셋이 애초에 고정이라 무엇을 빼든 똑같다), 분할 A·B 모두에서 같은
고정 적합을 평가만 다르게 적용한다. 방식 4(핀 비율)는 묶음1(NO1, 핀이 이동한 P1 구간)을
학습에서 항상 뺀다 — `묶음 1(P1)은 제외하고 맞춘다`(지시서).

입력: results/values_by_image.csv (scan_values.py 산출물)
산출물: results/calibration_loso.csv (방식 x 정의 x 사진선택 x 분할 요약, 규칙 10: 방식0이
항상 같은 그룹의 첫 줄), results/calibration_by_specimen.csv(단위별 상세, 참고).
"""
import os

import numpy as np
import pandas as pd

from calibration import (NORMAL_BUNDLES, PHYSICAL_SCALE, TEAMA_EVAL_BUNDLES, TEAMA_TRAIN_BUNDLES,
                          direction, evaluate_fold, fit_const, fit_fixed_scale, fit_linear, judge,
                          photo_select_images, setup_table, specimen_unit)

HERE = os.path.dirname(os.path.abspath(__file__))

DEFS = {
    "cross50": ("bearing_cross50_px", "pin_cross50_px"),
    "grad_peak": ("bearing_grad_peak_px", "pin_grad_peak_px"),
    "moment": ("bearing_moment_px", "pin_moment_px"),
}
PHOTO_SELECTIONS = ["all10", "last5", "no7excl"]
METHOD_FITTERS = {"0_평균찍기": fit_const, "2_전체선형회귀": fit_linear, "3_물리스케일고정": fit_fixed_scale}


def normal_units(setup: pd.DataFrame) -> list:
    return [u for u in setup.index if u == "NO1" or (u.startswith("B") and 2 <= int(u[1:]) <= 10)]


def ng_units(setup: pd.DataFrame) -> list:
    return [u for u in setup.index if u in ("B11", "B12")]


def loo_rows(setup: pd.DataFrame, fit_fn, exclude_from_train=frozenset()):
    """표준 LOO: normal_units 중 exclude_from_train에 없는 것들로 돌아가며 held-one-out.
    exclude_from_train에 있는 unit은 한 번도 학습에 못 들어가지만, (훈련 전체 집합)으로
    적합한 모델로 "평가만" 받는다(핀비율 방식의 묶음1처럼)."""
    units = normal_units(setup)
    trainable = [u for u in units if u not in exclude_from_train]
    rows = []
    for held in trainable:
        train = setup.loc[[u for u in trainable if u != held]]
        predict_fn, _ = fit_fn(train)
        rows += evaluate_fold(predict_fn, setup.loc[[held]])
    # exclude_from_train 유닛들: 전체(trainable) 학습 모델로 평가만
    if exclude_from_train:
        train_all = setup.loc[trainable]
        predict_fn, _ = fit_fn(train_all)
        held = [u for u in units if u in exclude_from_train]
        if held:
            rows += evaluate_fold(predict_fn, setup.loc[held])
    return rows


def ng_eval_rows(setup: pd.DataFrame, fit_fn, train_units: list):
    train = setup.loc[train_units]
    predict_fn, params = fit_fn(train)
    return evaluate_fold(predict_fn, setup.loc[ng_units(setup)]), params


def summarize(rows: list, method: str, definition: str, selection: str, split: str,
              baseline_mae: float, params: dict = None) -> dict:
    normal_rows = [r for r in rows if r["unit"] not in ("B11", "B12")]
    ng_rows = [r for r in rows if r["unit"] in ("B11", "B12")]
    mae = float(np.mean([r["abs_err_um"] for r in normal_rows])) if normal_rows else float("nan")
    mx = float(np.max([r["abs_err_um"] for r in normal_rows])) if normal_rows else float("nan")
    mis = sum(1 for r in normal_rows if r["판정"] == "NG")  # 정상품 오판정 수
    out = dict(정의=definition, 방식=method, 사진선택=selection, 분할=split,
               n_normal=len(normal_rows), MAE_um=mae, MAX_um=mx,
               평균찍기대비_um=(mae - baseline_mae) if not np.isnan(baseline_mae) else np.nan,
               정상오판정수=mis,
               # Codex adversarial-review 반영: 이 표의 모든 수치는 재장착 반복이 없는
               # 동일 장착(규칙 8) 안에서 나온 것이다 — 방법 선택의 최종 근거가 아니라
               # 탐색 결과로 읽어야 한다는 걸 CSV 자체에 남긴다(README뿐 아니라).
               검증_범위="동일장착내_재장착반복없음(규칙8)")
    for r in ng_rows:
        tag = "NG1" if r["unit"] == "B11" else "NG2"
        out[f"{tag}_판정"] = r["판정"]
        out[f"{tag}_방향"] = r["방향"]
        out[f"{tag}_pred_mm"] = round(r["pred_mm"], 4)
    if params:
        out["slope"] = params.get("slope")
        out["intercept"] = params.get("intercept")
    return out


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "values_by_image.csv"), encoding="utf-8-sig")

    summary_rows, detail_rows = [], []
    for def_name, (bcol, pcol) in DEFS.items():
        for sel in PHOTO_SELECTIONS:
            images = photo_select_images(df, sel)
            setup_b = setup_table(images, bcol)
            tmp = images.copy()
            tmp["ratio"] = tmp[bcol] / tmp[pcol]
            setup_ratio = setup_table(tmp, "ratio")

            # --- 분할 A: 시편 단위 LOO ---
            baseline_rows_a = loo_rows(setup_b, fit_const)
            ng_base_a, _ = ng_eval_rows(setup_b, fit_const, normal_units(setup_b))
            baseline_mae_a = float(np.mean([r["abs_err_um"] for r in baseline_rows_a]))

            for method, fitter in METHOD_FITTERS.items():
                rows = loo_rows(setup_b, fitter)
                ng_rows, params = ng_eval_rows(setup_b, fitter, normal_units(setup_b))
                summary_rows.append(summarize(rows + ng_rows, method, def_name, sel, "A_시편LOO",
                                               baseline_mae_a, params))
                detail_rows += [dict(**r, 정의=def_name, 방식=method, 사진선택=sel, 분할="A_시편LOO")
                                 for r in rows + ng_rows]

            # 방식 1: 묶음4·5·6 고정 학습(시편 단위 테이블에서 B4,B5,B6), 전체 normal+NG 평가
            # `predict_fn1`으로 이름을 분리했다(Codex adversarial-review 반영) — 아래 분할B
            # 블록의 `for method, fitter in METHOD_FITTERS.items()` 루프가 똑같은 이름
            # `predict_fn`을 마지막 메서드(3_물리스케일고정) 것으로 덮어쓰므로, 이름을 같이
            # 쓰면 분할B에서 방식1을 "재현"할 때 실제로는 다른 방식의 적합 함수를 쓰는
            # 새로운 버그가 생긴다 — 고치는 과정에서 스스로 발견해 피했다.
            fixed_train = setup_b.loc[["B4", "B5", "B6"]]
            predict_fn1, params1 = fit_linear(fixed_train)
            held = [u for u in normal_units(setup_b) if u not in ("B4", "B5", "B6")]
            rows1 = evaluate_fold(predict_fn1, setup_b.loc[held]) if held else []
            ng1 = evaluate_fold(predict_fn1, setup_b.loc[["B11", "B12"]])
            summary_rows.append(summarize(rows1 + ng1, "1_3점선형(NO4~6)", def_name, sel, "A_시편LOO",
                                           baseline_mae_a, params1))
            detail_rows += [dict(**r, 정의=def_name, 방식="1_3점선형(NO4~6)", 사진선택=sel, 분할="A_시편LOO")
                             for r in rows1 + ng1]

            # 방식 4: 핀 비율 정규화 후 선형, 묶음1(NO1)은 항상 학습 제외
            rows4 = loo_rows(setup_ratio, fit_linear, exclude_from_train={"NO1"})
            ng4, params4 = ng_eval_rows(setup_ratio, fit_linear, [u for u in normal_units(setup_ratio) if u != "NO1"])
            summary_rows.append(summarize(rows4 + ng4, "4_핀비율정규화선형", def_name, sel, "A_시편LOO",
                                           baseline_mae_a, params4))
            detail_rows += [dict(**r, 정의=def_name, 방식="4_핀비율정규화선형", 사진선택=sel, 분할="A_시편LOO")
                             for r in rows4 + ng4]

            # --- 분할 B: 팀원A 분할(묶음4~6 학습 -> 묶음7~10 예측) ---
            # "no7excl" 사진 선택은 묶음7을 풀에서 통째로 뺐으므로 setup_b에 B7이 아예 없다
            # — 있는 묶음만 평가 대상으로 쓴다(예: no7excl일 때는 B8~10만 평가).
            train_b = setup_b.loc[["B4", "B5", "B6"]]
            eval_units_b = [u for u in ("B" + str(x) for x in TEAMA_EVAL_BUNDLES) if u in setup_b.index]
            baseline_fn_b, _ = fit_const(train_b)
            baseline_rows_b = evaluate_fold(baseline_fn_b, setup_b.loc[eval_units_b])
            baseline_mae_b = float(np.mean([r["abs_err_um"] for r in baseline_rows_b]))

            for method, fitter in METHOD_FITTERS.items():
                predict_fn, params = fitter(train_b)
                rows_b = evaluate_fold(predict_fn, setup_b.loc[eval_units_b])
                ng_b = evaluate_fold(predict_fn, setup_b.loc[["B11", "B12"]])
                summary_rows.append(summarize(rows_b + ng_b, method, def_name, sel, "B_팀원A분할",
                                               baseline_mae_b, params))
                detail_rows += [dict(**r, 정의=def_name, 방식=method, 사진선택=sel, 분할="B_팀원A분할")
                                 for r in rows_b + ng_b]

            # 방식1 분할B: 학습(묶음4~6)은 분할A와 완전히 같아 `predict_fn1`(위에서 이미 적합)을
            # 재사용하지만, **평가 대상은 분할B 전용으로 다시 뽑는다**(Codex adversarial-review
            # 반영) — 전에는 분할A의 `rows1`(NO1·B2·B3·B7~10, 7개 정상 시편 전부)을 그대로
            # 재사용해서, "NO4~6 학습 -> NO7~10 평가"라는 분할B 정의와 다르게 NO1~3까지
            # 섞인 수치가 분할B 행으로 들어갔었다(실제 산출 CSV의 n_normal=7이 그 증거였다).
            rows1_b = evaluate_fold(predict_fn1, setup_b.loc[eval_units_b])
            summary_rows.append(summarize(rows1_b + ng1, "1_3점선형(NO4~6)", def_name, sel, "B_팀원A분할",
                                           baseline_mae_b, params1))
            detail_rows += [dict(**r, 정의=def_name, 방식="1_3점선형(NO4~6)", 사진선택=sel, 분할="B_팀원A분할")
                             for r in rows1_b + ng1]

            # 방식4 split B: ratio 테이블의 B4~6 학습, 핀 window_clipped는 묶음1에만 있어 영향 없음
            train_ratio_b = setup_ratio.loc[["B4", "B5", "B6"]]
            predict_fn4b, params4b = fit_linear(train_ratio_b)
            rows4b = evaluate_fold(predict_fn4b, setup_ratio.loc[eval_units_b])
            ng4b = evaluate_fold(predict_fn4b, setup_ratio.loc[["B11", "B12"]])
            summary_rows.append(summarize(rows4b + ng4b, "4_핀비율정규화선형", def_name, sel, "B_팀원A분할",
                                           baseline_mae_b, params4b))
            # 다른 모든 방식·분할 블록과 달리 이 블록만 detail_rows에 안 넣고 있었다(Codex
            # `/codex:review` 반영) — calibration_by_specimen.csv에 방식4/분할B 행이
            # 통째로 빠져 있었다. 요약(calibration_loso.csv)은 맞았지만 시편별 검증이 불가능했다.
            detail_rows += [dict(**r, 정의=def_name, 방식="4_핀비율정규화선형", 사진선택=sel, 분할="B_팀원A분할")
                             for r in rows4b + ng4b]

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(HERE, "results", "calibration_loso.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"저장: {summary_path} ({len(summary)}행)")

    detail = pd.DataFrame(detail_rows)
    detail_path = os.path.join(HERE, "results", "calibration_by_specimen.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"저장: {detail_path} ({len(detail)}행)")

    pd.set_option("display.width", 240)
    print("\nall10, A_시편LOO 기준 요약:")
    sub = summary[(summary["사진선택"] == "all10") & (summary["분할"] == "A_시편LOO")]
    print(sub[["정의", "방식", "MAE_um", "MAX_um", "평균찍기대비_um", "정상오판정수",
               "NG1_판정", "NG1_방향", "NG2_판정", "NG2_방향"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
