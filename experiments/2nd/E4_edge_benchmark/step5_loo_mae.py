"""E4 5단계((참고) 지표): 평균 찍기 기준선(규칙 10) 대비 정의별 LOO MAE. 핀 비율 환산
(D_mm = D_shaft_px / D_pin_px * 10, LOO 학습 셋업의 평균 치우침만 뺌 — 0.2b 팀원B 방식과
같다)과 물리 스케일 환산(D_mm = D_shaft_px * 0.0367, 마찬가지로 LOO 평균 치우침만 뺌)
둘 다 보인다. **이건 E6(캘리브레이션 비교)의 축소판 미리보기일 뿐, 이 실험에서 teacher를
고르는 기준으로 쓰지 않는다**(완료 기준). 정상 10개 셋업(묶음1~10, NO1~NO10) LOO,
NG1·NG2는 평가만 한다(규칙 4) — 여기서는 "LOO 학습에 쓰지 않는다"는 원칙만 지키고,
NG 평가 자체(OK/NG 판정)는 E6에서 다룬다.

입력: results/edge_method_benchmark.csv(베어링), cache/e4/pin_columns.csv(핀)
산출물: results/loo_mae_by_definition.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHYSICAL_SCALE = 0.0367  # mm/px, 0.2절
NORMAL_BUNDLES = list(range(1, 11))


def loo_mae(setup_vals: dict, true_vals: dict, convert):
    """setup_vals: {묶음: px}, true_vals: {묶음: mm}. convert(px)->mm 변환 후, 학습 9개
    셋업의 평균 치우침(실측-예측)만 빼고 held-out 1개를 예측한다."""
    bundles = list(setup_vals.keys())
    errs = []
    for held in bundles:
        train = [b for b in bundles if b != held]
        bias = np.mean([true_vals[b] - convert(setup_vals[b]) for b in train])
        pred = convert(setup_vals[held]) + bias
        errs.append(abs(true_vals[held] - pred) * 1000)  # mm -> um
    return float(np.mean(errs)), float(np.max(errs))


def baseline_mae(true_vals: dict):
    bundles = list(true_vals.keys())
    errs = []
    for held in bundles:
        train_mean = np.mean([true_vals[b] for b in bundles if b != held])
        errs.append(abs(true_vals[held] - train_mean) * 1000)
    return float(np.mean(errs)), float(np.max(errs))


def main():
    bench = pd.read_csv(os.path.join(HERE, "results", "edge_method_benchmark.csv"), encoding="utf-8-sig")
    pin_cols = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "pin_columns.csv"), encoding="utf-8-sig")

    bench10 = bench[bench["묶음"].isin(NORMAL_BUNDLES)]
    true_vals = bench10.groupby("묶음")["실측mm"].first().to_dict()

    rows = [dict(정의="(기준) 평균 찍기", 환산="-", n_setups=len(true_vals),
                 loo_mae_um=baseline_mae(true_vals)[0], loo_max_um=baseline_mae(true_vals)[1])]

    for name, g in bench10.groupby("정의"):
        shaft_px = g.groupby("묶음")["diameter_med"].median().to_dict()
        # esf2/esf3_outer는 셋업에 따라 ESF 적합이 통째로 실패해(단 붕괴, step1 참고) 10개
        # 전부가 아니라 일부 셋업만 들어간다 — 그래서 이 LOO는 다른 정의보다 "쉬운"(값이
        # 일관되게 나오는) 부분집합에서만 계산된다. n_setups로 그 사실을 남긴다(Codex
        # adversarial-review 반영 — 전엔 이 차이를 표에 안 남겨서 과소비교로 오해할 수 있었다).
        tv_this = {b: true_vals[b] for b in shaft_px}

        rows.append(dict(정의=name, 환산="물리스케일(0.0367mm/px)", n_setups=len(shaft_px),
                          **dict(zip(("loo_mae_um", "loo_max_um"),
                                     loo_mae(shaft_px, tv_this, lambda p: p * PHYSICAL_SCALE)))))

    pin_defs = {"cross50": ("top_cross", "bot_cross"), "grad_peak": ("top_peak", "bot_peak"),
                "moment": ("top_moment", "bot_moment"), "esf1": ("top_esf1", "bot_esf1")}
    for name, (tcol, bcol) in pin_defs.items():
        pin_cols["diam_tmp"] = pin_cols[bcol] - pin_cols[tcol]
        pin_px = pin_cols[pin_cols["묶음"].isin(NORMAL_BUNDLES)].groupby("묶음")["diam_tmp"].median()
        bench_this = bench10[bench10["정의"] == name]
        if bench_this.empty or pin_px.empty:
            continue
        shaft_px = bench_this.groupby("묶음")["diameter_med"].median()
        common = sorted(set(shaft_px.index) & set(pin_px.index))
        ratio_vals = {b: float(shaft_px[b] / pin_px[b] * 10) for b in common}
        tv = {b: true_vals[b] for b in common}
        rows.append(dict(정의=name, 환산="핀비율(D/Dpin*10)", n_setups=len(common),
                          **dict(zip(("loo_mae_um", "loo_max_um"), loo_mae(ratio_vals, tv, lambda p: p)))))

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "loo_mae_by_definition.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(out.round(2).to_string(index=False))
    print("\n평균 찍기를 확실히 이기는 정의는 없다 — 이 미리보기 수준에서는(참고, E6에서 정식 비교).")
    print("주의: esf2/esf3_outer는 n_setups가 10보다 작다(ESF 적합이 실패하는 셋업은 통째로 빠짐) "
          "— 다른 정의(n_setups=10)와 LOO MAE를 직접 비교하면 안 된다(서로 다른 부분집합).")


if __name__ == "__main__":
    main()
