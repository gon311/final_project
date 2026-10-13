"""E4 4단계((참고) 지표): 정의별 셋업 중앙값 지름(px)과 실측(mm)의 Spearman + 순열 p
(규칙 13, n=13 전체 셋업 — NG1·NG2 포함, 정상만이면 n=10도 같이 보인다).

**이 결과로 teacher를 고르지 않는다**(지시서 완료 기준: "실측 상관이나 MAE로 고르지
않는다. n=13이고, 정답을 보고 고른 최선도 우연 수준이었다"). 참고 수치로만 둔다.

입력: results/edge_method_benchmark.csv (step1 산출물)
산출물: results/spearman_by_definition.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261013
N_PERM = 5000


def spearman_perm(x, y, seed, n_perm=N_PERM):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(r)).mean())
    return r, p


def main():
    bench = pd.read_csv(os.path.join(HERE, "results", "edge_method_benchmark.csv"), encoding="utf-8-sig")

    rows = []
    for name, g in bench.groupby("정의"):
        g13 = g.sort_values("묶음")
        r13, p13 = spearman_perm(g13["diameter_med"].to_numpy(), g13["실측mm"].to_numpy(), SEED)
        g10 = g13[g13["묶음"].between(1, 10)]
        r10, p10 = spearman_perm(g10["diameter_med"].to_numpy(), g10["실측mm"].to_numpy(), SEED + 1)
        rows.append(dict(정의=name, spearman_r_n13=r13, perm_p_n13=p13,
                          spearman_r_n10_정상만=r10, perm_p_n10_정상만=p10))
    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "spearman_by_definition.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(out.round(3).to_string(index=False))
    print("\n(참고용 수치다 — teacher 선택 기준으로 쓰지 않는다. n=13/10이고 우연 수준의 r도 "
          "흔하다, 지시서 완료 기준·0.2b 팀원B ceiling 결과와 같은 이유.)")


if __name__ == "__main__":
    main()
