"""E13-4: E11-A 하단 재계산 (PREREG 8절, 사후 민감도). E11 split-half·고정 패턴·판정 함수를 그대로 쓴다. 결론 문장에 쓰지 않는다."""
import pandas as pd
from paths import RESULTS
import e13_common as C

E = C._e11  # E11 모듈: analyze/spearman_perm/COMBOS 등 (수정 없음)
e, m, first_c = C.load()
base = E.edges[(E.edges["단계"] == "본시험(암막)") & E.edges.specimen.isin(C.SPECS)]
rough = pd.read_csv(C.ROUGH).set_index("specimen_1021")


def windowed(win_fn, trim):
    """시편별 창(1번 이미지 c 기준, 10장 동일) 안 열만 남기고, 이미지마다 양 끝 trim열 제외."""
    parts = []
    for s, g in base.groupby("specimen"):
        lo, hi = win_fn(first_c[s])
        for _, gi in g[(g.x >= lo) & (g.x <= hi)].groupby("filename"):
            gi = gi.sort_values("x")
            parts.append(gi.iloc[trim:len(gi) - trim] if trim else gi)
    return pd.concat(parts)


by_rows, sum_rows = [], []
for wname, wfn in [("C", C.window_C), ("C_rev", C.window_Crev)]:
    for trim in (0, 3):
        E.main = windowed(wfn, trim)  # analyze/build가 모듈 전역 main을 읽는다
        df, _, _ = E.analyze("y_bot", 2)
        df["Rq"] = rough["Rq"].reindex(df.specimen).to_numpy()
        by_rows.append(df.assign(window=wname, trim=trim))
        n_has = int(df.has_specific.sum())
        for rv in ("Rz", "Rq"):
            rho, p = E.spearman_perm(df[rv].to_numpy(), df["a_spec_px"].to_numpy())
            stage1 = n_has >= 6
            case = 3 if not stage1 else (1 if (rho > 0 and p < 0.05) else 2)
            sum_rows.append(dict(window=wname, trim=trim, rough_var=rv, n_spec=len(df), n_has_specific=n_has,
                                 n_cols_median=int(df.n_cols.median()), rho=rho, p_one_sided=p, conclusion_case=case, post_hoc=True))
pd.concat(by_rows).to_csv(RESULTS / "E13_4_by_specimen.csv", index=False)
S = pd.DataFrame(sum_rows); S.to_csv(RESULTS / "E13_4_summary.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(3).to_string(index=False))
d = pd.concat(by_rows); d = d[(d.window == "C") & (d.trim == 0)]
print(d[["specimen", "n_cols", "interpolated", "S_within_med", "S_cross_med", "n_cross_pairs", "a_spec_um", "p05", "has_specific", "Rz", "Rq"]].round(3).to_string(index=False))
