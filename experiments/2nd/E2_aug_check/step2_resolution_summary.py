"""E2 해상도 조건 요약. results/resolution_check.csv를 조건×격자위상별로 집계한다."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "resolution_check.csv"), encoding="utf-8-sig")
    ok = df[df["ok"]]
    print(f"측정 성공: {len(ok)}/{len(df)}")

    agg = ok.groupby(["condition", "phase"]).agg(
        n=("file", "count"),
        med_diam_cross50_px=("d_diam_cross50_px", lambda s: s.abs().median()),
        max_diam_cross50_px=("d_diam_cross50_px", lambda s: s.abs().max()),
        med_diam_grad_peak_px=("d_diam_grad_peak_px", lambda s: s.abs().median()),
        max_diam_grad_peak_px=("d_diam_grad_peak_px", lambda s: s.abs().max()),
        med_step_transition_frac=("bot_step_transition_frac", "median"),
        med_n_cols_detected=("n_cols_detected", "median"),
    )
    # 셋업(묶음)별 worst-case: 각 묶음의 중앙값 중 가장 큰 값(Codex adversarial-review 반영
    # — scan을 130장 전체로 넓힌 것과 같은 이유, 전체 중앙값만으로는 특정 셋업의 최악을 놓친다)
    worst = ok.groupby(["condition", "phase", "묶음"])["d_diam_cross50_px"].apply(lambda s: s.abs().median())
    worst_by_cond = worst.groupby(["condition", "phase"]).max().rename("worst_setup_diam_cross50_med")
    worst_bundle = worst.groupby(["condition", "phase"]).idxmax().apply(lambda t: t[2]).rename("worst_setup_묶음")
    agg = agg.join(worst_by_cond).join(worst_bundle)

    out_path = os.path.join(HERE, "results", "resolution_summary.csv")
    agg.to_csv(out_path, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"\n저장: {out_path}\n")
    print(agg.round(4).to_string())

    phys_scale = 0.0367  # mm/px, 0.2절(참고 환산일 뿐 — 캘리브레이션 아님, E6 참고)
    print("\n물리 스케일 참고 환산(0.0367mm/px, 캘리브레이션 아님):")
    print((agg["med_diam_cross50_px"] * phys_scale * 1000).round(1).rename("med_um").to_string())


if __name__ == "__main__":
    main()
