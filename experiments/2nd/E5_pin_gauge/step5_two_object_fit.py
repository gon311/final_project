"""E5 5단계: 2-물체 모델 D_px = D_mm / s + c (s = mm/px, c = 엣지 오프셋 px)를 F3 셋업에
맞춘다. 핀(10.000mm, 알려진 값)과 샤프트(분류표 실측mm)를 같은 회귀에 넣고, s = 1/기울기를
물리 스케일(0.0367mm/px)과 비교한다. **폰 렌즈가 비텔레센트릭이라 핀과 샤프트는 카메라
이동·초점 변화의 영향을 다르게 받으므로, 이 계산은 "같은 평면이라면"이라는 가설 검증용이다.**

**NG1·NG2는 적합에 쓰지 않는다**(0.5절 규칙 4: "NG1·NG2를 학습이나 하이퍼파라미터 튜닝에
쓰지 않는다. 평가에만 쓴다"). 정상 F3 시편(NO7~NO10, NO1 재촬영)만으로 s·c를 적합하고,
NG1·NG2는 그 식으로 **held-out 예측**만 한다(샤프트 px로 mm을 역산해 실측mm과 비교).
NG를 적합에 포함한 예전 결과는 `sensitivity_with_ng`에 참고용으로만 남긴다
(Codex 3차 adversarial-review 반영).

**부트스트랩 CI는 "검증 근거"가 아니라 기술통계다.** 각 묶음이 핀 1점(항상 10.000mm 고정)과
샤프트 1점(같은 사진에서 찍은, 묶음별로 거의 같은 촬영 조건)으로 축약되므로, 묶음을
재표집해도 "2개 물체 뭉치"라는 구조 자체는 거의 안 바뀐다. 이 CI는 독립 재장착·깊이·카메라·
초점 조건의 불확실성을 반영하지 않는다 — README "한계" 참고.

입력: cache/pin_bearing_scalars.csv, results/pin_position_compare.csv
산출물: results/two_object_fit.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache")
PHYSICAL_SCALE_MM_PER_PX = 0.0367
SEED = 20261013
N_BOOT = 5000

PIN_MM = 10.000
NORMAL_F3_BUNDLES = [7, 8, 9, 10, 13]  # NO7, NO8, NO9, NO10, NO1(재촬영) — NG1·NG2 제외
NG_BUNDLES = {11: "NG1", 12: "NG2"}
ALL_F3_BUNDLES = sorted(set(NORMAL_F3_BUNDLES) | set(NG_BUNDLES))


def fit_points(points):
    """points: list of (D_mm, D_px). 최소제곱 D_px = beta1*D_mm + beta0. s=1/beta1, c=beta0."""
    d_mm = np.array([p[0] for p in points], dtype=float)
    d_px = np.array([p[1] for p in points], dtype=float)
    beta1, beta0 = np.polyfit(d_mm, d_px, 1)
    return float(1.0 / beta1), float(beta0)


def bundle_pin_px(bundle_df):
    """묶음의 핀 px 평균. window_clipped=True(창이 xR에 잘린) 프레임은 뺀다."""
    clean = bundle_df[~bundle_df["window_clipped"]]
    use = clean if len(clean) else bundle_df  # 전부 clipped면 어쩔 수 없이 전체를 쓴다
    return use["method_c_pin_relative"].mean(), len(bundle_df) - len(clean)


def bundle_points(bundle_df):
    """묶음 하나 -> [(핀mm, 핀px), (샤프트mm, 샤프트px)], 제외한 clipped 프레임 수."""
    pin_px, n_clipped_excluded = bundle_pin_px(bundle_df)
    shaft_px = bundle_df["shaft_d_line"].mean()
    shaft_mm = bundle_df["실측mm"].astype(float).mean()
    return [(PIN_MM, pin_px), (shaft_mm, shaft_px)], n_clipped_excluded


def fit_config(bundles_data, bundle_ids, rng):
    points = []
    n_clipped_excluded_total = 0
    for g in bundle_ids:
        pts, n_clipped = bundle_points(bundles_data[g])
        points += pts
        n_clipped_excluded_total += n_clipped
    s_hat, c_hat = fit_points(points)

    boot_s, boot_c = [], []
    for _ in range(N_BOOT):
        sample_ids = rng.choice(bundle_ids, size=len(bundle_ids), replace=True)
        pts = []
        for g in sample_ids:
            p, _ = bundle_points(bundles_data[g])
            pts += p
        try:
            s_b, c_b = fit_points(pts)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            continue
        boot_s.append(s_b)
        boot_c.append(c_b)
    boot_s, boot_c = np.array(boot_s), np.array(boot_c)
    return dict(
        bundle_ids=list(bundle_ids), n_points=len(points),
        n_window_clipped_frames_excluded=n_clipped_excluded_total,
        s_mm_per_px=s_hat,
        s_bootstrap_spread_p2_5_p97_5=[float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
        c_px=c_hat,
        c_bootstrap_spread_p2_5_p97_5=[float(np.percentile(boot_c, 2.5)), float(np.percentile(boot_c, 97.5))],
        s_vs_physical_scale_ratio=float(s_hat / PHYSICAL_SCALE_MM_PER_PX),
    )


def predict_ng_heldout(bundles_data, fitted):
    """정상 시편만으로 적합한 s·c로 NG1·NG2 샤프트 mm을 역산해 held-out 평가한다
    (적합에는 전혀 쓰지 않음 — 0.5절 규칙 4)."""
    s_hat, c_hat = fitted["s_mm_per_px"], fitted["c_px"]
    out = {}
    for g, name in NG_BUNDLES.items():
        d = bundles_data[g]
        shaft_px = d["shaft_d_line"].mean()
        actual_mm = d["실측mm"].astype(float).mean()
        predicted_mm = s_hat * (shaft_px - c_hat)
        err_um = (predicted_mm - actual_mm) * 1000
        out[name] = dict(
            bundle=g, actual_mm=float(actual_mm), predicted_mm=float(predicted_mm),
            error_um=float(err_um),
            actual_direction="하한 아래(NG)" if actual_mm < 20.010 else ("상한 위(NG)" if actual_mm > 20.030 else "OK"),
            predicted_direction="하한 아래" if predicted_mm < 20.010 else ("상한 위" if predicted_mm > 20.030 else "OK"),
        )
    return out


def main():
    scal = pd.read_csv(os.path.join(CACHE_DIR, "pin_bearing_scalars.csv"), encoding="utf-8-sig")
    pos = pd.read_csv(os.path.join(HERE, "results", "pin_position_compare.csv"), encoding="utf-8-sig")
    df = scal.merge(pos[["file", "method_c_pin_relative", "window_clipped"]], on="file")

    bundles_data = {g: d for g, d in df.groupby("묶음")}
    rng = np.random.default_rng(SEED)

    primary_with_no7 = fit_config(bundles_data, NORMAL_F3_BUNDLES, rng)
    primary_without_no7 = fit_config(bundles_data, [g for g in NORMAL_F3_BUNDLES if g != 7], rng)

    result = dict(
        physical_scale_mm_per_px=PHYSICAL_SCALE_MM_PER_PX,
        pin_mm=PIN_MM,
        note="D_px = D_mm/s + c. 핀(10.000mm)과 샤프트(분류표 실측mm)를 같은 회귀에 넣었다. "
             "비텔레센트릭이 아닌 폰 렌즈 때문에 핀·샤프트가 카메라 이동·초점 변화를 다르게 받을 수 있어 "
             "'같은 평면이라면'이라는 가정이 깔린 가설 검증이다.",
        caveat_on_spread="s_bootstrap_spread_*는 검증용 95% 신뢰구간이 아니라 기술통계다. "
             "핀 mm은 항상 10.000으로 고정이고 같은 묶음의 핀·샤프트 점은 같은 촬영 조건을 공유하므로, "
             "묶음을 재표집해도 '핀 뭉치 vs 샤프트 뭉치' 2군집 구조 자체는 거의 안 바뀐다. "
             "독립 재장착·깊이·카메라·초점 불확실성은 전혀 반영하지 않는다 — README 한계 참고.",
        seed=SEED, n_bootstrap=N_BOOT,
        normal_bundles_used_for_fit=NORMAL_F3_BUNDLES,
        ng_bundles_held_out=list(NG_BUNDLES.values()),
        primary_fit_normal_only=dict(with_NO7=primary_with_no7, without_NO7=primary_without_no7),
        ng_heldout_prediction=dict(
            with_NO7_fit=predict_ng_heldout(bundles_data, primary_with_no7),
            without_NO7_fit=predict_ng_heldout(bundles_data, primary_without_no7),
        ),
        sensitivity_with_ng=dict(
            note="NG1·NG2를 적합 데이터에 포함한 경우(0.5절 규칙4 위반, 참고/민감도 분석 전용 — 결론 근거로 쓰지 않음)",
            with_NO7=fit_config(bundles_data, ALL_F3_BUNDLES, rng),
            without_NO7=fit_config(bundles_data, [g for g in ALL_F3_BUNDLES if g != 7], rng),
        ),
    )

    out_path = os.path.join(HERE, "results", "two_object_fit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
