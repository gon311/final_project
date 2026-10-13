"""E4 1단계: 정의별·셋업별 안정성 지표를 뽑는다.

비교 정의(지시서 1~4항):
  - cross50 (top_cross/bot_cross), grad_peak (top_peak/bot_peak): `edge_subpx` 그대로.
  - moment: `moment_centroid`(그래디언트 1차 모멘트).
  - esf1: ESF 1단(top은 항상 이 값을 쓴다 — 위 경계는 절벽 하나).
  - esf2_outer, esf3_outer: 하단 ESF 2·3단 중 **바깥 단**(배경에 가장 가까운 단, 팀원B
    M2~M5 "바깥 단" 계열과 같은 방향)을 하단 위치로 쓴다. 위는 여전히 esf1.

**ESF 적합 실패(`*_ok=False`, `scan_definitions.py`가 NaN으로 기록)는 지름·σ·잔차 계산에서
전부 뺀다**(Codex adversarial-review 반영 — 전에는 호출부가 `success`를 안 보고 실패값도
그대로 집계에 넣었다). `fit_ok_frac` 열에 이미지별 유효 열 비율을 같이 남겨서, 설령 한
이미지에 유효 열이 하나도 없어 그 이미지가 통째로 빠지더라도(esf2/esf3은 묶음 7~10·12에서
실제로 이런 일이 생긴다, README 참고) 몇 장이 빠졌는지 추적할 수 있게 했다.

지표:
  - 셋업 내 지름(px) σ: 열별 지름(bot-top, 유효한 열만) 중앙값을 이미지당 하나로 모은 뒤
    셋업 내 이미지 간 표준편차. (esf2/esf3은 유효 이미지 수가 10장보다 적을 수 있다.)
  - 열 방향 잔차 σ: 이미지별로 top·bot 열 위치(유효한 열만)에 강건 직선적합을 하고 남는
    잔차의 표준편차(상·하단 평균), 셋업 내 중앙값.
  - 엣지 정의 간 체계적 차이: cross50 대비 median(지름_정의 - 지름_cross50)(유효 이미지만).

산출물: results/edge_method_benchmark.csv (정의 x 셋업), results/esf_fit_failure_rate.csv
(정의 x 셋업 ESF 적합 실패율), 콘솔에 NO7 포함/제외 요약.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

BOTTOM_DEFS = {
    "cross50": ("top_cross", "bot_cross", None),
    "grad_peak": ("top_peak", "bot_peak", None),
    "moment": ("top_moment", "bot_moment", None),
    "esf1": ("top_esf1", "bot_esf1", ("top_esf1_ok", "bot_esf1_ok")),
    "esf2_outer": ("top_esf1", "bot_esf2_outer", ("top_esf1_ok", "bot_esf2_ok")),
    "esf3_outer": ("top_esf1", "bot_esf3_outer", ("top_esf1_ok", "bot_esf3_ok")),
}


def robust_fit_resid(x, y):
    p = np.polyfit(x, y, 1)
    r = y - np.polyval(p, x)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    k = np.abs(r) < 4 * 1.4826 * mad
    if k.sum() < 2:
        return r
    p2 = np.polyfit(x[k], y[k], 1)
    return y - np.polyval(p2, x)


def per_image_stats(df, top_col, bot_col, ok_cols):
    out, fail_rows = [], []
    for fn, g in df.groupby("file"):
        g = g.sort_values("x")
        n_total = len(g)
        if ok_cols is not None:
            ok = g[ok_cols[0]].to_numpy(bool) & g[ok_cols[1]].to_numpy(bool)
        else:
            ok = np.ones(n_total, dtype=bool)
        row0 = g.iloc[0]
        fail_rows.append(dict(file=fn, 묶음=row0["묶음"], n_total=n_total, n_ok=int(ok.sum())))

        gg = g[ok]
        x = gg["x"].to_numpy(dtype=float)
        top, bot = gg[top_col].to_numpy(dtype=float), gg[bot_col].to_numpy(dtype=float)
        if len(x) < 4:
            continue  # 유효 열이 강건 직선적합을 하기에 너무 적다(ESF2/3이 이미지 전체에서 실패한 경우 포함)
        diam_med = float(np.median(bot - top))
        rt = robust_fit_resid(x, top)
        rb = robust_fit_resid(x, bot)
        resid_std = float(np.std(np.concatenate([rt, rb])))
        out.append(dict(file=fn, 묶음=row0["묶음"], 시편=row0["시편"], 실측mm=row0["실측mm"], 초점=row0["초점"],
                         diameter_px=diam_med, resid_std=resid_std, fit_ok_frac=ok.mean()))
    return pd.DataFrame(out), pd.DataFrame(fail_rows)


def main():
    cols = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "bearing_columns.csv"), encoding="utf-8-sig")

    cross_img, _ = per_image_stats(cols, *BOTTOM_DEFS["cross50"])
    cross_img = cross_img.set_index("file")

    rows, fail_rows = [], []
    for name, (tcol, bcol, ok_cols) in BOTTOM_DEFS.items():
        img, fail = per_image_stats(cols, tcol, bcol, ok_cols)
        fail["정의"] = name
        fail_rows.append(fail)
        if img.empty:
            print(f"  (경고) {name}: 모든 이미지가 유효 열 부족으로 빠졌다 — 셋업 집계 불가")
            continue
        diff_vs_cross = (img.set_index("file")["diameter_px"] - cross_img["diameter_px"]).dropna()
        setup_agg = img.groupby("묶음").agg(
            시편=("시편", "first"), 실측mm=("실측mm", "first"), 초점=("초점", "first"),
            n_valid_images=("file", "count"), diameter_med=("diameter_px", "median"),
            setup_sigma=("diameter_px", "std"), resid_std_med=("resid_std", "median"),
        )
        setup_agg["정의"] = name
        setup_agg["systematic_diff_vs_cross50_overall_med"] = diff_vs_cross.median() if len(diff_vs_cross) else np.nan
        rows.append(setup_agg.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out = out[["정의", "묶음", "시편", "초점", "실측mm", "n_valid_images", "diameter_med", "setup_sigma",
               "resid_std_med", "systematic_diff_vs_cross50_overall_med"]]
    out_path = os.path.join(HERE, "results", "edge_method_benchmark.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path} ({len(out)}행)\n")

    fail_all = pd.concat(fail_rows, ignore_index=True)
    fail_summary = fail_all.groupby(["정의", "묶음"]).apply(
        lambda g: pd.Series(dict(n_images=len(g), col_fail_rate=1 - g["n_ok"].sum() / g["n_total"].sum())),
        include_groups=False)
    fail_path = os.path.join(HERE, "results", "esf_fit_failure_rate.csv")
    fail_summary.reset_index().to_csv(fail_path, index=False, encoding="utf-8-sig")
    print(f"저장: {fail_path} (ESF 적합 실패율, 정의 x 셋업 — 열 단위, '단 붕괴' 포함)\n")

    pd.set_option("display.width", 220)
    print("정의별 셋업 내 σ(px) 요약 (13개 셋업 median, 규칙 12: NO7=묶음7 포함/제외):")
    for name in BOTTOM_DEFS:
        sub = out[out["정의"] == name]
        if sub.empty:
            print(f"  {name:12s}: 유효 데이터 없음")
            continue
        all13 = sub["setup_sigma"].median()
        excl7 = sub[sub["묶음"] != 7]["setup_sigma"].median()
        diff_med = sub["systematic_diff_vs_cross50_overall_med"].iloc[0]
        resid_med = sub["resid_std_med"].median()
        n_setups = len(sub)
        print(f"  {name:12s}: σ_med({n_setups}셋업)={all13:.4f}px  σ_med(NO7제외)={excl7:.4f}px  "
              f"잔차σ_med={resid_med:.4f}px  cross50 대비 체계차={diff_med:+.4f}px")


if __name__ == "__main__":
    main()
