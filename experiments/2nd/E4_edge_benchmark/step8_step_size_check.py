"""E4 한계 검증(Codex adversarial-review 반영): 베어링부 스캔을 `step=10`으로 거칠게 한
선택이 cross50뿐 아니라 **moment·ESF 순위에도** 유효한지 `step=2`(130장 전체, `scan_definitions.py
--step 2`) 재스캔으로 교차검증한다. 전에는 cross50만 `eda_measure_raw.csv`(원본, step=2)와
대조했다 — moment·ESF는 그래디언트 모양·극댓값 위치에 직접 의존해서 성긴 샘플링이 공간적으로
특정 구간을 건너뛰면 σ를 왜곡할 수 있다는 지적을 받았다.

`step1_summary.py`의 `per_image_stats`/`BOTTOM_DEFS`를 그대로 재사용해(중복 구현 방지) 같은
정의를 `step=2` 데이터에 적용하고, `step=10` 결과와 정의별 σ·순위를 나란히 둔다.

산출물: results/step_size_sensitivity.csv
"""
import os

import pandas as pd

from step1_summary import BOTTOM_DEFS, per_image_stats

HERE = os.path.dirname(os.path.abspath(__file__))


def sigma_table(cols: pd.DataFrame) -> dict:
    out = {}
    for name, (tcol, bcol, ok_cols) in BOTTOM_DEFS.items():
        img, _ = per_image_stats(cols, tcol, bcol, ok_cols)
        if img.empty:
            out[name] = dict(sigma_med=float("nan"), n_setups=0)
            continue
        setup_sigma = img.groupby("묶음")["diameter_px"].std()
        out[name] = dict(sigma_med=float(setup_sigma.median()), n_setups=int(setup_sigma.notna().sum()))
    return out


def main():
    step10 = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "bearing_columns.csv"), encoding="utf-8-sig")
    step2 = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "bearing_columns_step2.csv"), encoding="utf-8-sig")

    s10 = sigma_table(step10)
    s2 = sigma_table(step2)

    rows = []
    for name in BOTTOM_DEFS:
        rows.append(dict(정의=name,
                          sigma_step10=s10[name]["sigma_med"], n_setups_step10=s10[name]["n_setups"],
                          sigma_step2=s2[name]["sigma_med"], n_setups_step2=s2[name]["n_setups"]))
    out = pd.DataFrame(rows)
    out["차이_step10_minus_step2"] = out["sigma_step10"] - out["sigma_step2"]

    # 정의 간 "순위"가 step 크기와 무관하게 보존되는지(안정성 기준 cross50·esf1·moment가
    # 계속 grad_peak·esf2·esf3보다 낮은 σ를 유지하는지)
    rank10 = out.sort_values("sigma_step10")["정의"].tolist()
    rank2 = out.dropna(subset=["sigma_step2"]).sort_values("sigma_step2")["정의"].tolist()

    out_path = os.path.join(HERE, "results", "step_size_sensitivity.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(out.round(4).to_string(index=False))
    print(f"\nstep=10 순위(σ 오름차순): {rank10}")
    print(f"step=2  순위(σ 오름차순): {rank2}")
    print(f"상위 3(cross50/esf1/moment 그룹) vs 하위(grad_peak/esf2/esf3 그룹) 순서가 "
          f"두 step 크기에서 같은가: {rank10[:3] == rank2[:3] if len(rank2) >= 3 else '비교 불가(esf2/esf3 n 부족)'}")


if __name__ == "__main__":
    main()
