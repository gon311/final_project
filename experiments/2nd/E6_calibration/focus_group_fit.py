"""E6 방식5((참고) 초점 구간별로 따로 맞추기). F1/F2/F3 안에서만 전체 선형회귀(방식2와
같은 적합)를 해 그 구간 안의 셋업끼리 잔차가 얼마나 작아지는지 본다. 구간당 셋업이 2~4개뿐
이라(F1: 묶음1~4, F2: 묶음5~6, F3: 묶음7~10·13) 제대로 된 LOO는 의미가 없다 — 참고용으로
**구간 안 적합·구간 안 잔차(사실상 in-sample)**만 본다, 그래서 다른 방식들과 직접 МAE
비교는 하지 않는다(지시서가 이 항목을 "(참고)"로만 표시한 이유와 같다).

**이 분석은 "시편(specimen_unit)"이 아니라 "묶음(shoot)" 단위로 묶는다**(Codex `/codex:review`
반영). 다른 모든 E6 스크립트는 NO1(묶음1·13)을 같은 시편이라고 합쳐서 다루지만(캘리브레이션
학습·평가에는 그게 맞다), **초점은 셋업(묶음)의 속성이지 시편의 속성이 아니다** — NO1은
묶음1(F1)과 묶음13(F3)이라는 서로 다른 초점 조건에서 두 번 찍힌 시편이라, 시편 단위로
합치면 "그 시편의 초점이 뭐냐"는 질문 자체가 성립하지 않는다(이전 버전은 `묶음<=10` 필터로
묶음13을 통째로 걸러서 이 문제를 우회했는데, 그 바람에 F3 그룹에 NO1의 F3 촬영분이 아예
빠지는 결과를 낳았다 — Codex가 지적한 문제). 묶음 단위로 묶으면 묶음1과 묶음13이 각각
독립된 데이터점으로 F1·F3에 올바르게 들어간다.

입력: results/values_by_image.csv
산출물: results/focus_group_fit.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFS = {"cross50": "bearing_cross50_px", "grad_peak": "bearing_grad_peak_px", "moment": "bearing_moment_px"}
NG_BUNDLES = {11, 12}


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "values_by_image.csv"), encoding="utf-8-sig")
    df = df[~df["묶음"].isin(NG_BUNDLES)]  # NG만 뺀다(참고용 적합이라 NG 평가는 생략) — 묶음13은 그대로 둔다

    rows = []
    for def_name, col in DEFS.items():
        setup = df.groupby("묶음").agg(실측mm=("실측mm", "first"), px=(col, "median"), 초점=("초점", "first"))
        for focus, g in setup.groupby("초점"):
            if len(g) < 2 or g["px"].std() == 0:
                for bundle, r in g.iterrows():
                    rows.append(dict(정의=def_name, 초점=focus, 묶음=int(bundle), n_in_group=len(g),
                                      실측mm=r["실측mm"], pred_mm=np.nan, err_um=np.nan))
                continue
            a, b = np.polyfit(g["px"].to_numpy(), g["실측mm"].to_numpy(), 1)
            for bundle, r in g.iterrows():
                pred = a * r["px"] + b
                rows.append(dict(정의=def_name, 초점=focus, 묶음=int(bundle), n_in_group=len(g),
                                  실측mm=r["실측mm"], pred_mm=pred, err_um=(pred - r["실측mm"]) * 1000,
                                  slope=a, intercept=b))

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "focus_group_fit.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    summary = out.groupby(["정의", "초점"]).agg(n=("묶음", "count"),
                                                in_sample_MAE_um=("err_um", lambda s: s.abs().mean()))
    print(summary.round(2).to_string())
    print("\n(참고용: 구간 안 묶음이 2~4개뿐이라 in-sample 잔차다 — LOO가 아니다. 다른 방식의 "
          "MAE와 직접 비교하지 않는다, 지시서 (참고) 표기와 같은 이유. NO1(묶음1·13)은 F1·F3에 "
          "각각 독립 데이터점으로 들어간다 — 초점은 시편이 아니라 묶음의 속성이라서다.)")


if __name__ == "__main__":
    main()
