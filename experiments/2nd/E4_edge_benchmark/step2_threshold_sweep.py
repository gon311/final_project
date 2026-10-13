"""E4 2단계(지시서 비교 정의 5): 하단 문턱 sweep. 상단은 cross50(50%)으로 고정하고
하단만 10~90%(9단계)로 바꿔 지름을 다시 잰다. **teacher 후보를 고르는 용도가 아니라
"하단 문턱에 따라 무엇이 얼마나 바뀌는가"를 보는 축**이다(지시서 원문, 완료 기준 참고).

같이 뽑는 "하단 단 전환 비율"(지시서 지표 3번)은 각 열의 ESF 3단 적합 결과(inner/mid/outer,
`scan_definitions.py`가 이미 뽑아 둔 값)를 "단 후보 위치"로 쓰고, 문턱이 10%에서 90%로
바뀔 때 하단 교차가 가리키는 가장 가까운 단이 달라지는 열의 비율로 정의한다 — 팀원B가
관찰한 "50% 문턱이 셋업마다 다른 단에 걸린다"(0.2b)를 "문턱을 실제로 움직여 보면 얼마나
자주 걸리는 단이 바뀌는가"로 정량화한 것이다.

산출물: results/bottom_threshold_sweep.csv (셋업 x 문턱 중앙값 지름 + Spearman/순열p),
results/bottom_step_transition_by_setup.csv.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
SEED = 20261013
N_PERM = 5000


def spearman_perm(x, y, seed, n_perm=N_PERM):
    def rank(a):
        return pd.Series(a).rank().to_numpy()

    rx, ry = rank(x), rank(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(r)).mean())
    return r, p


def main():
    cols = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "bearing_columns.csv"), encoding="utf-8-sig")

    # --- 문턱 sweep: 셋업별 median 지름 (상단 cross50 고정) ---
    sweep_rows = []
    for t in THRESH_LEVELS:
        col = f"bot_thresh{t}"
        cols["diam_tmp"] = cols[col] - cols["top_cross"]
        per_img = cols.groupby("file").agg(묶음=("묶음", "first"), 시편=("시편", "first"),
                                            실측mm=("실측mm", "first"), 초점=("초점", "first"),
                                            diam=("diam_tmp", "median"))
        setup = per_img.groupby("묶음").agg(시편=("시편", "first"), 실측mm=("실측mm", "first"),
                                            초점=("초점", "first"), diameter_med=("diam", "median"),
                                            n=("diam", "count"))
        setup["threshold_pct"] = t
        sweep_rows.append(setup.reset_index())
    sweep = pd.concat(sweep_rows, ignore_index=True)

    # 정상 10개 셋업(묶음1~10, NO1~NO10) 기준 Spearman + 순열 p (참고용, teacher 선택에 안 씀)
    normal = sweep[sweep["묶음"].between(1, 10)]
    corr_rows = []
    for t in THRESH_LEVELS:
        sub = normal[normal["threshold_pct"] == t].sort_values("묶음")
        r, p = spearman_perm(sub["diameter_med"].to_numpy(), sub["실측mm"].to_numpy(), seed=SEED)
        corr_rows.append(dict(threshold_pct=t, spearman_r=r, perm_p=p))
    corr = pd.DataFrame(corr_rows)
    sweep = sweep.merge(corr, on="threshold_pct", how="left")

    out_path = os.path.join(HERE, "results", "bottom_threshold_sweep.csv")
    sweep.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(sweep)}행)\n")
    print("문턱별 Spearman r (정상 10개 셋업, 실측 셔플 5000회 순열 p, seed=%d):" % SEED)
    print(corr.round(3).to_string(index=False))
    print(f"\nr 범위: {corr['spearman_r'].min():.3f} ~ {corr['spearman_r'].max():.3f}"
          f" (팀원B 0.2b 인용: -0.47~+0.59, 81조합·정답 보고 고른 최선조차 우연 수준)")

    # --- 단 전환 비율: ESF3 inner/mid/outer를 "단 후보"로, 40% vs 60% 문턱(50% 바로 양옆,
    # 실제 측정이 쓰는 지점 근처의 "흔들림" 폭)에서 가장 가까운 단이 바뀌는 열의 비율.
    # 10% vs 90%처럼 양 끝을 비교하면 애초에 "가장 가까운 단"이 다른 게 당연해(항상 1.0이
    # 나와 의미가 없다 — 처음 시도에서 확인한 실패, 아래 "한계"에도 적었다) 50% 바로 옆에서
    # 보는 쪽으로 바꿨다. 이게 팀원B가 말한 "50% 문턱이 첫 작은 단 꼭대기와 거의 같은 높이라
    # 셋업마다 다른 단에 걸린다"(0.2b)를 재는 축에 더 가깝다.
    # ESF3 적합이 실패한(단 붕괴, bot_esf3_ok=False) 열은 "단 후보" 자체가 NaN이라 뺀다
    # (Codex adversarial-review 반영 — 전에는 실패값도 그대로 "가장 가까운 단" 계산에 섞였다).
    valid = cols["bot_esf3_ok"].to_numpy(bool)
    sub = cols[valid].copy()
    centers = sub[["bot_esf3_inner", "bot_esf3_mid", "bot_esf3_outer"]].to_numpy()

    def nearest_step(cross_vals, centers):
        d = np.abs(cross_vals[:, None] - centers)
        return np.argmin(d, axis=1)

    step_lo = nearest_step(sub["bot_thresh40"].to_numpy(), centers)
    step_hi = nearest_step(sub["bot_thresh60"].to_numpy(), centers)
    sub["step_transition"] = step_lo != step_hi
    trans = sub.groupby("묶음").agg(시편=("시편", "first"), 실측mm=("실측mm", "first"),
                                    n_cols=("x", "count"), transition_frac=("step_transition", "mean"))
    n_excluded_by_bundle = cols.groupby("묶음")["bot_esf3_ok"].apply(lambda s: int((~s).sum()))
    trans["n_esf3_실패_제외"] = n_excluded_by_bundle
    trans_path = os.path.join(HERE, "results", "bottom_step_transition_by_setup.csv")
    trans.to_csv(trans_path, encoding="utf-8-sig")
    print(f"\n저장: {trans_path}")
    print(f"ESF3 적합 성공(단 붕괴 아님) 열만 썼다 — 묶음별 n_cols: {trans['n_cols'].min()}~{trans['n_cols'].max()}"
          f" (묶음7~10·12는 ESF3가 거의 항상 실패해 n_cols가 0에 가깝거나 행 자체가 빠질 수 있다)")
    print(f"단 전환 비율(40%->60% 문턱 사이 가장 가까운 ESF3 단이 바뀌는 열 비율), 셋업별 범위: "
          f"{trans['transition_frac'].min():.3f}~{trans['transition_frac'].max():.3f}")


if __name__ == "__main__":
    main()
