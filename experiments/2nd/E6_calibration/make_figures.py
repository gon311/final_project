"""E6 그림 2장. (1) 방식별 LOO MAE 막대그래프(all10, 분할A, 정의 3종) — 평균 찍기 기준선을
가로줄로 같이 그려 "이기는 조합이 있는가"를 한눈에 보게 한다. (2) cross50 px vs 실측mm
산점도 + 방식2(전체선형)·방식3(물리스케일고정) 적합선 — 상관이 약하다는 걸 시각으로 보인다."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab.plotting import setup_korean_font  # noqa: E402
from calibration import setup_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


WARNING_KO = "주의: 동일 장착 10장 반복 결과다 — 재장착 반복 검증 없음(규칙 8). 방법 선택 근거가 아니라 탐색 결과."
WARNING_EN = "Caution: single-mount repeats only, no remount validation (rule 8). Exploratory, not a method recommendation."


def main():
    font_ok = setup_korean_font()
    summary = pd.read_csv(os.path.join(HERE, "results", "calibration_loso.csv"), encoding="utf-8-sig")
    sub = summary[(summary["사진선택"] == "all10") & (summary["분할"] == "A_시편LOO")]
    repeat = pd.read_csv(os.path.join(HERE, "results", "within_mount_repeatability.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    defs = sub["정의"].unique()
    methods = sub["방식"].unique()
    x = np.arange(len(methods))
    width = 0.25
    for i, d in enumerate(defs):
        vals = [sub[(sub["정의"] == d) & (sub["방식"] == m)]["MAE_um"].iloc[0] for m in methods]
        ax.bar(x + (i - 1) * width, vals, width, label=d)
    baseline = sub[sub["방식"] == "0_평균찍기"]["MAE_um"].iloc[0]
    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"평균 찍기 기준선 ({baseline:.1f}um)" if font_ok else f"baseline ({baseline:.1f}um)")
    # Codex adversarial-review 반영: 같은 장착 10장 반복성(위 repeatability_check.py)을 음영
    # 밴드로 같이 보여준다 — 방식 간 MAE 차이가 이 밴드보다 작으면 "방법 차이"가 아니라
    # "촬영 노이즈 안"일 수 있다는 걸 그림 자체에서 바로 비교하게 한다.
    sigma_lo, sigma_hi = repeat["median_setup_sigma_um_물리스케일환산"].min(), repeat["median_setup_sigma_um_물리스케일환산"].max()
    ax.axhspan(sigma_lo, sigma_hi, color="gray", alpha=0.15,
               label=f"동일장착 10장 반복성 참고대({sigma_lo:.1f}~{sigma_hi:.1f}um)" if font_ok
               else f"single-mount repeatability band ({sigma_lo:.1f}-{sigma_hi:.1f}um)")
    ax.set_xticks(x, methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("LOO MAE (um)" if font_ok else "LOO MAE (um)")
    ax.set_title("방식별 LOO MAE (사진선택=all10, 분할=A 시편LOO)"
                  if font_ok else "LOO MAE by calibration method")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.01, WARNING_KO if font_ok else WARNING_EN, ha="center", fontsize=7.5, color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(HERE, "figures", "mae_by_method.png"), dpi=140)
    plt.close(fig)

    df = pd.read_csv(os.path.join(HERE, "results", "values_by_image.csv"), encoding="utf-8-sig")
    setup = setup_table(df, "bearing_cross50_px")
    normal = setup.loc[[u for u in setup.index if u not in ("B11", "B12")]]
    ng = setup.loc[[u for u in ("B11", "B12") if u in setup.index]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normal["px"], normal["실측mm"], s=30, c="C0",
               label="정상 10개(학습에 씀)" if font_ok else "normal (used to fit)")
    ax.scatter(ng["px"], ng["실측mm"], s=40, c="red", marker="x",
               label="NG1·NG2(평가만, 학습 제외 — 규칙4)" if font_ok else "NG (eval only)")
    for u, r in setup.iterrows():
        ax.annotate(u, (r["px"], r["실측mm"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(setup["px"].min() - 1, setup["px"].max() + 1, 50)
    # 방식2·3 모두 정상 10개로만 맞춘다(규칙4: NG는 학습에 쓰지 않는다) — NG는 위에 평가
    # 참고용으로만 찍는다.
    a2, b2 = np.polyfit(normal["px"], normal["실측mm"], 1)
    ax.plot(xs, a2 * xs + b2, "C1-", label=f"방식2 전체선형(정상10, {a2*1000:+.1f}um/px)" if font_ok else "linear fit")
    b3 = float((normal["실측mm"] - 0.0367 * normal["px"]).mean())
    ax.plot(xs, 0.0367 * xs + b3, "C2--", label="방식3 물리스케일고정(0.0367mm/px)" if font_ok else "fixed scale")
    ax.set_xlabel("cross50 지름 (px, 셋업 중앙값)" if font_ok else "cross50 diameter (px)")
    ax.set_ylabel("실측 (mm)" if font_ok else "measured (mm)")
    ax.set_title("cross50 px vs 실측mm (정상 10개로 적합, NG는 평가만)"
                  if font_ok else "cross50 px vs measured mm (fit on normal only)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, WARNING_KO if font_ok else WARNING_EN, ha="center", fontsize=7.5, color="firebrick")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(HERE, "figures", "cross50_px_vs_mm.png"), dpi=140)
    plt.close(fig)
    print("저장: figures/mae_by_method.png, figures/cross50_px_vs_mm.png")


if __name__ == "__main__":
    main()
