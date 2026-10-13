"""E6 캘리브레이션 비교 공용 엔진. 방식(0~4) x 정의(cross50/grad_peak/moment) x 사진 선택
(all10/last5/no7excl) x 평가 분할(A LOO/B 팀원A)을 전부 같은 함수로 돌린다 — 다음 촬영
데이터에도 그대로 다시 쓸 수 있게 입력을 분류표+측정값 DataFrame 하나로 일반화했다
(완료 기준 요구사항).

**단위:** 셋업(시편) 단위로만 학습한다(규칙 3) — 이미지 10장을 그대로 회귀에 넣으면 같은
셋업의 반복 프레임이 과하게 영향을 미친다. 사진 선택 변형은 "셋업 대표 px를 구할 때 어느
프레임을 쓰는가"에서만 작동하고, 그 이후(회귀 적합)는 항상 셋업 1점이다.

**NO1 중복(0.3절):** 같은 시편 NO1이 묶음1(P1, 핀 이동 전)과 묶음13(NO1 재촬영, P2)에 둘
있다. 시편 단위 LOO에서는 둘을 하나의 "NO1" 단위로 합쳐서(이미지 합쳐 대표 px를 내고)
함께 빼거나 함께 평가한다.
"""
import numpy as np
import pandas as pd

NORMAL_BUNDLES = list(range(1, 11))
NO1_BUNDLES = {1, 13}  # 같은 시편, 시편 단위 LOO에서 함께 취급(0.3절)
NG_BUNDLES = {11: "NG1", 12: "NG2"}
TEAMA_TRAIN_BUNDLES = (4, 5, 6)
TEAMA_EVAL_BUNDLES = (7, 8, 9, 10)
PHYSICAL_SCALE = 0.0367  # mm/px, 0.2절
TOL_LO, TOL_HI = 20.010, 20.030  # mm, 0.3절 공차


def photo_select_images(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """셋업 대표 px를 낼 때 쓸 이미지 부분집합을 고른다(묶음별로 적용).
    (i) all10: 그대로. (ii) last5: 묶음 안에서 시각 순 마지막 5장(재초점 직후 흐름 완화,
    팀원B 11.1µm). (iii) no7excl: 묶음7(NO7, 재초점 직후) 전체를 풀에서 뺀다."""
    if variant == "all10":
        return df
    if variant == "last5":
        out = []
        for g, sub in df.groupby("묶음"):
            out.append(sub.sort_values("시각").tail(5))
        return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]
    if variant == "no7excl":
        return df[df["묶음"] != 7]
    raise ValueError(f"unknown photo_select variant: {variant!r}")


def specimen_unit(bundle: float) -> str:
    """시편 단위 LOO의 "단위 이름"을 돌려준다. NO1(묶음1·13)은 같은 단위로 합친다."""
    if int(bundle) in NO1_BUNDLES:
        return "NO1"
    return f"B{int(bundle)}"


def setup_table(df: pd.DataFrame, px_col: str) -> pd.DataFrame:
    """이미지 단위 df -> 시편(단위) 단위 표. index=unit, columns=[묶음들(참고), 실측mm, px]."""
    d = df.copy()
    d["unit"] = d["묶음"].map(specimen_unit)
    agg = d.groupby("unit").agg(실측mm=("실측mm", "first"), px=(px_col, "median"),
                                 n_images=(px_col, "count"), 묶음목록=("묶음", lambda s: sorted(set(s))))
    return agg


def fit_const(train: pd.DataFrame):
    mean_mm = float(train["실측mm"].mean())
    return lambda px: mean_mm, dict(slope=np.nan, intercept=mean_mm)


def fit_linear(train: pd.DataFrame):
    if len(train) < 2 or train["px"].std() == 0:
        return fit_const(train)
    a, b = np.polyfit(train["px"].to_numpy(), train["실측mm"].to_numpy(), 1)
    return lambda px: float(a * px + b), dict(slope=float(a), intercept=float(b))


def fit_fixed_scale(train: pd.DataFrame):
    b = float((train["실측mm"] - PHYSICAL_SCALE * train["px"]).mean())
    return lambda px: float(PHYSICAL_SCALE * px + b), dict(slope=PHYSICAL_SCALE, intercept=b)


def judge(mm: float) -> str:
    return "OK" if TOL_LO <= mm <= TOL_HI else "NG"


def direction(mm: float) -> str:
    if mm < TOL_LO:
        return "하한 아래"
    if mm > TOL_HI:
        return "상한 위"
    return "공차 안"


def evaluate_fold(predict_fn, held: pd.DataFrame) -> list:
    rows = []
    for unit, r in held.iterrows():
        pred = predict_fn(r["px"])
        err_um = (pred - r["실측mm"]) * 1000
        rows.append(dict(unit=unit, 실측mm=r["실측mm"], pred_mm=pred, err_um=err_um,
                          abs_err_um=abs(err_um), 판정=judge(pred), 방향=direction(pred)))
    return rows


METHODS = {
    "0_평균찍기": fit_const,
    "2_전체선형회귀": fit_linear,
    "3_물리스케일고정": fit_fixed_scale,
}  # 1(3점)과 4(핀비율)는 train 구성이 달라 run_calibration.py에서 직접 다룬다
