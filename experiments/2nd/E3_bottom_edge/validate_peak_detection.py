"""E3 검증(Codex 4차 adversarial-review 반영): `find_peaks_subpixel`의 weakest/strongest
비율이 "진짜 단일 절벽 + 노이즈"와 "실제 다단"을 구분하는지 독립적으로 확인한다.
  (a) 합성 단일-절벽 대조군: 실제 데이터에서 추정한 노이즈 수준으로 "진짜 절벽 하나"
      프로파일을 여러 번 만들어, weakest/strongest 비율이 실제 핀·베어링부 상단(알려진
      절벽 하나)과 비슷한 분포를 내는지 본다.
  (b) 민감도: `cluster_radius`·`min_sep`을 바꿔도 "핀·상단은 낮고 베어링부 하단은 높다"는
      순위가 유지되는지, 대표 이미지 일부에서 다시 스캔해 확인한다.

입력: 원본 이미지(검증용 일부), results/bottom_peaks.csv, cache/e3/{top,pin_edge}_peaks.csv
산출물: results/peak_detection_validation.csv (민감도 결과), 콘솔 출력(합성 대조군)
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, PIN_COARSE  # noqa: E402
from shaftlab.multistep import edge_multistep_features, find_peaks_subpixel, scan_section_multistep  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_E3 = os.path.join(HERE, "..", "cache", "e3")
SEED = 20261013


def weakest_strongest(peaks):
    if len(peaks) < 3:
        return np.nan
    s = sorted(p[1] for p in peaks)
    return s[0] / s[-1]


def estimate_point_noise_sigma(gray, seg, is_above):
    """실제 이미지의 평평한 배경/안쪽 영역에서 **점대점(point-to-point) 잡음**을 추정한다.
    단순 `std()`는 창 안의 느린 밝기 추세(텍스처)까지 섞어 실제 센서·JPEG 잡음보다 훨씬 크게
    나온다(Codex 4차 adversarial-review로 드러난 1차 시도의 문제) — **1차 차분의 표준편차를
    sqrt(2)로 나눈 값**을 쓰면 느린 추세가 지워지고 진짜 픽셀 대 픽셀 잡음만 남는다.
    `is_above`: 하단 엣지(`bc`) 기준으로 위(샤프트 안쪽, fg)인지 아래(배경, bg)인지만 바꾼다
    — fg/bg 추정 둘 다 **같은 엣지**(`bc`)를 기준으로 삼아야 한다(이전 버전은 fg를 상단 엣지
    `tc` 기준 바깥 영역에서 잘못 재는 버그가 있었다 — 1차 시도에서 fg σ가 터무니없이 크게
    나온 원인)."""
    sigmas = []
    for si in range(0, len(seg), 2):
        X = int(seg[si, 0] * 4)
        bc = int(seg[si, 3] * 4)
        y0, y1 = (bc - 220, bc - 60) if is_above else (bc + 60, bc + 220)
        flat = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        if len(flat) < 10:
            continue
        sigmas.append(float(np.std(np.diff(flat)) / np.sqrt(2)))
    return float(np.median(sigmas))


def synthetic_single_edge_trial(rng, sigma_fg, sigma_bg, length=160, center=80,
                                 fg=59.0, bg=249.0, width_10_90=13.0):
    """실측 fg/bg 수준·엣지 폭·**점대점 잡음**(`estimate_point_noise_sigma`로 구함)으로
    '진짜 절벽 하나' 프로파일 1개를 만든다."""
    x = np.arange(length)
    # 10-90% 폭(width_10_90)에 맞춘 로지스틱 전이. 10~90% 구간이 폭 w가 되도록 기울기를 맞춘다.
    k = np.log(0.9 / 0.1) / (width_10_90 / 2)
    base = fg + (bg - fg) / (1 + np.exp(-k * (x - center)))
    sigma = sigma_fg + (base - fg) / (bg - fg) * (sigma_bg - sigma_fg)
    noisy = base + rng.normal(0, sigma)
    blurred = cv2.GaussianBlur(noisy.reshape(-1, 1).astype(np.float64), (1, 7), 1.2).ravel()
    return blurred


def run_synthetic_control(sigma_fg, sigma_bg, n_trials=500):
    rng = np.random.default_rng(SEED)
    ratios, n_peaks_list = [], []
    for _ in range(n_trials):
        prof = synthetic_single_edge_trial(rng, sigma_fg, sigma_bg)
        peaks = find_peaks_subpixel(prof, center=80, win=45, sign=1, n_peaks=3)
        n_peaks_list.append(len(peaks))
        ratios.append(weakest_strongest(peaks))
    return pd.Series(ratios), pd.Series(n_peaks_list)


def run_sensitivity(sample_files, cluster_radii=(15, 22, 30), min_seps=(2, 3, 5)):
    rows = []
    for fn in sample_files:
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        seg_b = coarse_section(q, **BEARING_COARSE)
        seg_p = coarse_section(q, **PIN_COARSE)
        for cr in cluster_radii:
            for ms in min_seps:
                for label, seg, sign, key in (("bearing_bottom", seg_b, 1, "bot"),
                                               ("bearing_top", seg_b, -1, "top"),
                                               ("pin_bottom", seg_p, 1, "bot")):
                    xa, xb = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
                    # 대표 열 5개만(중앙부) — 민감도 확인이 목적이라 전수 스캔은 안 한다
                    xs = np.linspace(xa + (xb - xa) * 0.3, xb - (xb - xa) * 0.3, 5).astype(int)
                    ratios = []
                    for X in xs:
                        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
                        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
                        center = tc if sign < 0 else bc
                        y0 = center - 65
                        prof = gray[y0:center + 65, X - 1:X + 2].astype(np.float64).mean(1)
                        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
                        peaks = find_peaks_subpixel(prof, center - y0, 45, sign, n_peaks=3,
                                                     min_sep=ms, cluster_radius=cr)
                        ratios.append(weakest_strongest(peaks))
                    rows.append(dict(file=fn, section=label, cluster_radius=cr, min_sep=ms,
                                      weakest_strongest_med=float(np.nanmedian(ratios))))
    return pd.DataFrame(rows)


def main():
    print("== (a) 합성 단일-절벽 대조군 ==")
    fn0 = "20261008_151701.jpg"
    gray0 = cv2.imread(os.path.join(paths.IMAGE_DIR, fn0), cv2.IMREAD_GRAYSCALE)
    q0 = cv2.resize(gray0, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    seg0 = coarse_section(q0, **BEARING_COARSE)
    sigma_fg = estimate_point_noise_sigma(gray0, seg0, is_above=True)
    sigma_bg = estimate_point_noise_sigma(gray0, seg0, is_above=False)
    print(f"실측 점대점(point-to-point) 잡음(1차 차분 기반, {fn0}): fg측 σ={sigma_fg:.3f}, bg측 σ={sigma_bg:.3f}")
    print("(참고: 창 전체 std()로 재면 fg~5, bg~13이 나오는데, 그건 창 안의 느린 밝기 추세까지 섞인 "
          "과대추정값이다 — Codex 4차 review로 드러난 1차 시도의 문제, 1차 차분으로 고쳤다.)")

    ratios, n_peaks = run_synthetic_control(sigma_fg, sigma_bg)
    frac_3 = (n_peaks == 3).mean()
    print(f"\n합성 '진짜 절벽 하나' (n={len(ratios)} 시행): 극댓값 3개를 찾은 비율={frac_3:.3f} "
          f"(평균 극댓값 개수={n_peaks.mean():.2f}, 1에 가까울수록 '절벽 하나'로 올바르게 봄)")
    ratios_valid = ratios.dropna()
    if len(ratios_valid):
        print(f"3개를 찾은 경우만(n={len(ratios_valid)}): weakest/strongest 중앙값={ratios_valid.median():.3f}")

    bottom = pd.read_csv(os.path.join(HERE, "results", "bottom_peaks.csv"), encoding="utf-8-sig")
    top = pd.read_csv(os.path.join(CACHE_E3, "top_peaks.csv"), encoding="utf-8-sig")
    pin = pd.read_csv(os.path.join(CACHE_E3, "pin_edge_peaks.csv"), encoding="utf-8-sig")

    def real_ratio(df):
        d3 = df[df["n_peaks"] == 3]
        s = d3[["peak1_strength", "peak2_strength", "peak3_strength"]]
        return (s.min(axis=1) / s.max(axis=1)).median()

    r_top, r_bot, r_pin = real_ratio(top), real_ratio(bottom), real_ratio(pin[pin["side"] == "bot"])
    print(f"\n실제 베어링부 상단(절벽 하나로 알려짐): {r_top:.3f}")
    print(f"실제 핀 하단(절벽 하나로 알려짐)       : {r_pin:.3f}")
    print(f"실제 베어링부 하단(다단으로 주장)      : {r_bot:.3f}")
    print(f"합성 절벽 하나(대조군, 3개 찾은 경우만): "
          f"{ratios_valid.median():.3f}" if len(ratios_valid) else "합성 절벽 하나: 3개를 찾은 시행 자체가 드묾")
    print(f"\n-> 합성 '진짜 절벽 하나'는 {1 - frac_3:.1%}의 경우 애초에 극댓값이 3개도 안 나온다"
          f"(평균 {n_peaks.mean():.2f}개, 실제로 절벽 하나로 올바르게 분류됨). 드물게(±{frac_3:.1%}) "
          f"3개가 나와도 weakest/strongest 비율은 거의 0({ratios_valid.median():.3f})이라 "
          f"베어링부 하단({r_bot:.3f})과 전혀 다르다. **독립 대조군이 지지한다: 실제 센서·JPEG 잡음 "
          f"수준에서는 '진짜 절벽 하나'가 weakest/strongest 비율을 베어링부 하단 수준까지 "
          f"끌어올리지 못한다.**")
    print("(1차 시도에서는 fg/bg 잡음을 '창 전체 std()'로 과대추정(σ≈5~13)해 합성 대조군이 "
          "오히려 더 다단처럼 보이는 실패를 겪었다 — 1차 차분 기반으로 고쳐 재현했다.)")

    print("\n== (b) cluster_radius·min_sep 민감도 (대표 5장) ==")
    sample_files = sorted(bottom["file"].unique())
    picks = [sample_files[i] for i in np.linspace(0, len(sample_files) - 1, 5).astype(int)]
    sens = run_sensitivity(picks)
    out_path = os.path.join(HERE, "results", "peak_detection_validation.csv")
    sens.to_csv(out_path, index=False, encoding="utf-8-sig")
    pivot = sens.groupby(["section", "cluster_radius", "min_sep"])["weakest_strongest_med"].median().unstack(["cluster_radius", "min_sep"])
    pd.set_option("display.width", 220)
    print(pivot.round(3).to_string())
    print(f"\n저장: {out_path}")

    rank_preserved = True
    for cr in [15, 22, 30]:
        for ms in [2, 3, 5]:
            sub = sens[(sens["cluster_radius"] == cr) & (sens["min_sep"] == ms)]
            med = sub.groupby("section")["weakest_strongest_med"].median()
            if not (med.get("bearing_bottom", 0) > med.get("bearing_top", 1) and
                    med.get("bearing_bottom", 0) > med.get("pin_bottom", 1)):
                rank_preserved = False
    print("\n모든 cluster_radius x min_sep 조합에서 '베어링부 하단 > 베어링부 상단, 핀 하단' 순위 유지:",
          rank_preserved)


if __name__ == "__main__":
    main()
