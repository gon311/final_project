"""E1 재현 확인:
  (1) shaftlab.measure_bearing이 기존 eda_measure_raw.csv의 B_d_line, B_d_peak_median을,
  (2) shaftlab.measure_pin이 기존 pin_gauge_raw.csv의 P_d_line, P_d_peak를
그대로 재현하는지 확인한다. 기대 파일 집합·행 수·오류(실패) 상태가 하나라도 원본과
다르거나 |Δ| >= 1e-6px인 항목이 있으면 비정상 종료(exit code 1)한다 (Codex 리뷰 반영).

사용: (experiments/2nd/.venv 활성화 후) python3 reproduce.py
산출물: results/reproduce_diff.csv, results/extract_profiles_check.csv
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import extract_profiles, measure_bearing, measure_pin, paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF_TOL_PX = 1e-6


def fail(problems, message):
    print(f"[FAIL] {message}")
    problems.append(message)


def report_and_exit(problems):
    """FAIL 보고서를 찍고 즉시 비정상 종료한다. 파일 집합 불일치처럼 이후 단계를
    돌리면 안 되는(예: KeyError) 문제도 여기서 바로 끝낸다 — 교집합만 추려서
    계속 진행하지 않는다. 불일치 자체가 실패다(Codex 3차 리뷰 반영)."""
    print(f"\n완료 기준(|Δ|<{DIFF_TOL_PX:.0e}px, 파일 집합·오류 상태 일치): FAIL ({len(problems)}건)")
    for p in problems:
        print(" -", p)
    sys.exit(1)


def all_finite(*values):
    """모든 값이 유한수(NaN/inf 아님)인지 확인한다. 새 측정값뿐 아니라 기존 CSV의
    기준값도 이걸로 걸러서, 어느 한쪽이 예외 없이 NaN을 내도 '성공'으로 잘못
    분류되지 않게 한다 (Codex 2차 리뷰 P2 반영)."""
    return all(np.isfinite(v) for v in values)


def check_file_sets(problems):
    """파일 집합·실패 집합이 기존 CSV와 정확히 같은지 강제한다."""
    bearing_orig = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    pin_orig = pd.read_csv(paths.PIN_GAUGE_RAW_CSV, encoding="utf-8-sig").set_index("file")

    files = sorted(glob.glob(os.path.join(paths.IMAGE_DIR, "*.jpg")))
    file_set = {os.path.basename(p) for p in files}
    expected_files = set(bearing_orig.index)

    if file_set != expected_files:
        fail(problems, f"이미지 폴더 파일 집합이 eda_measure_raw.csv와 다름: "
                        f"missing={expected_files - file_set}, extra={file_set - expected_files}")

    known_bearing_failures = set(bearing_orig.index[bearing_orig["B_d_line"].isna()])
    if known_bearing_failures != paths.EXCLUDED_FILES:
        fail(problems, f"기존 CSV의 베어링 측정 실패 파일 집합이 paths.EXCLUDED_FILES와 다름: "
                        f"csv_failures={known_bearing_failures}, EXCLUDED_FILES={paths.EXCLUDED_FILES}")

    # 원본 eda_pin_gauge.py는 raw[raw['시편'].notna()]로 핀 대상 파일을 거른다(0.3절:
    # 분류표에 없는 메모 사진만 빠짐). 같은 기준으로 기대 핀 파일 집합을 계산한다.
    expected_pin_files = set(bearing_orig.index[bearing_orig["시편"].notna()])
    actual_pin_files = set(pin_orig.index)
    if actual_pin_files != expected_pin_files:
        fail(problems, f"pin_gauge_raw.csv의 파일 집합이 '시편 notna' 기준과 다름: "
                        f"csv={actual_pin_files - expected_pin_files} (초과), "
                        f"누락={expected_pin_files - actual_pin_files}")

    return bearing_orig, pin_orig, sorted(file_set), known_bearing_failures, expected_pin_files


def check_extract_profiles_coverage(sample_files, problems, n_samples=3):
    """extract_profiles가 measure_section과 같은 열 수를 내는지 확인한다
    (2px 간격 중복 적용 버그 재발 방지, Codex 리뷰 반영)."""
    rows = []
    checked = 0
    for fn in sample_files:
        if checked >= n_samples:
            break
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        try:
            section = measure_bearing(gray)
        except Exception:
            continue
        profiles = extract_profiles(gray, section)
        n_cols = section["n_cols"]
        n_profiles = len(profiles)
        tol = max(2, round(0.05 * n_cols))
        ok = abs(n_profiles - n_cols) <= tol
        rows.append(dict(file=fn, n_cols=n_cols, n_profiles=n_profiles, tol=tol, ok=ok))
        if not ok:
            fail(problems, f"extract_profiles 열 수 불일치: {fn} n_cols={n_cols} n_profiles={n_profiles} (tol={tol})")
        checked += 1

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "results", "extract_profiles_check.csv"), index=False, encoding="utf-8-sig")
    print(f"\nextract_profiles 열 수 커버리지 점검 ({len(rows)}장): \n{out.to_string(index=False)}")


def measure_one(fn, gray, bearing_orig, pin_orig, known_bearing_failures, expected_pin_files, problems):
    row = {"file": fn}

    row["orig_B_d_line"] = bearing_orig.loc[fn, "B_d_line"]
    row["orig_B_d_peak_median"] = bearing_orig.loc[fn, "B_d_peak_median"]
    orig_bearing_failed = not all_finite(row["orig_B_d_line"], row["orig_B_d_peak_median"])

    try:
        r_line = measure_bearing(gray, edge_method="cross50")
        r_peak = measure_bearing(gray, edge_method="grad_peak")
        row["new_B_d_line"] = r_line["diameter_px"]
        row["new_B_d_peak_median"] = r_peak["diameter_px"]
        new_bearing_failed = not all_finite(row["new_B_d_line"], row["new_B_d_peak_median"])
        row["bearing_error"] = "" if not new_bearing_failed else "non-finite diameter_px (NaN/inf, 예외 없이 발생)"
    except Exception as e:  # noqa: BLE001
        row["new_B_d_line"] = np.nan
        row["new_B_d_peak_median"] = np.nan
        row["bearing_error"] = str(e)
        new_bearing_failed = True

    if new_bearing_failed != orig_bearing_failed:
        fail(problems, f"{fn}: 베어링 측정 성공/실패 상태가 기존 CSV와 다름 "
                        f"(orig_failed={orig_bearing_failed}, new_failed={new_bearing_failed})")
    if orig_bearing_failed != (fn in known_bearing_failures):
        fail(problems, f"{fn}: isfinite 기준 실패 판정이 known_bearing_failures(B_d_line만 본 집합)와 다름")

    if not orig_bearing_failed and not new_bearing_failed:
        row["diff_d_line"] = abs(row["new_B_d_line"] - row["orig_B_d_line"])
        row["diff_d_peak_median"] = abs(row["new_B_d_peak_median"] - row["orig_B_d_peak_median"])
        if row["diff_d_line"] >= DIFF_TOL_PX:
            fail(problems, f"{fn}: |diff_d_line|={row['diff_d_line']:.3e}px >= {DIFF_TOL_PX:.0e}")
        if row["diff_d_peak_median"] >= DIFF_TOL_PX:
            fail(problems, f"{fn}: |diff_d_peak_median|={row['diff_d_peak_median']:.3e}px >= {DIFF_TOL_PX:.0e}")
    else:
        row["diff_d_line"] = np.nan
        row["diff_d_peak_median"] = np.nan

    if fn in expected_pin_files:
        row["orig_P_d_line"] = pin_orig.loc[fn, "P_d_line"]
        row["orig_P_d_peak"] = pin_orig.loc[fn, "P_d_peak"]
        orig_pin_failed = not all_finite(row["orig_P_d_line"], row["orig_P_d_peak"])

        try:
            p_line = measure_pin(gray, edge_method="cross50")
            p_peak = measure_pin(gray, edge_method="grad_peak")
            row["new_P_d_line"] = p_line["diameter_px"]
            row["new_P_d_peak"] = p_peak["diameter_px"]
            new_pin_failed = not all_finite(row["new_P_d_line"], row["new_P_d_peak"])
            row["pin_error"] = "" if not new_pin_failed else "non-finite diameter_px (NaN/inf, 예외 없이 발생)"
        except Exception as e:  # noqa: BLE001
            row["new_P_d_line"] = np.nan
            row["new_P_d_peak"] = np.nan
            row["pin_error"] = str(e)
            new_pin_failed = True

        if new_pin_failed != orig_pin_failed:
            fail(problems, f"{fn}: 핀 측정 성공/실패 상태가 기존 CSV와 다름 "
                            f"(orig_failed={orig_pin_failed}, new_failed={new_pin_failed})")

        if not orig_pin_failed and not new_pin_failed:
            row["diff_P_d_line"] = abs(row["new_P_d_line"] - row["orig_P_d_line"])
            row["diff_P_d_peak"] = abs(row["new_P_d_peak"] - row["orig_P_d_peak"])
            if row["diff_P_d_line"] >= DIFF_TOL_PX:
                fail(problems, f"{fn}: |diff_P_d_line|={row['diff_P_d_line']:.3e}px >= {DIFF_TOL_PX:.0e}")
            if row["diff_P_d_peak"] >= DIFF_TOL_PX:
                fail(problems, f"{fn}: |diff_P_d_peak|={row['diff_P_d_peak']:.3e}px >= {DIFF_TOL_PX:.0e}")
        else:
            row["diff_P_d_line"] = np.nan
            row["diff_P_d_peak"] = np.nan
    else:
        for k in ("new_P_d_line", "new_P_d_peak", "orig_P_d_line", "orig_P_d_peak", "diff_P_d_line", "diff_P_d_peak"):
            row[k] = np.nan
        row["pin_error"] = "NOT_EXPECTED (분류표에 시편 정보 없음, 0.3절)"

    return row


def main():
    problems = []
    bearing_orig, pin_orig, files, known_bearing_failures, expected_pin_files = check_file_sets(problems)

    if problems:
        # 파일 집합·실패 집합이 기존 CSV와 다르면 이후 measure_one()이 bearing_orig.loc[fn]에서
        # KeyError로 죽는다(이미지 폴더에만 있는 파일 등). 교집합만 추려 계속하지 않고,
        # 불일치 자체를 제어된 실패로 보고하고 바로 끝낸다.
        report_and_exit(problems)

    rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        row = measure_one(fn, gray, bearing_orig, pin_orig, known_bearing_failures, expected_pin_files, problems)
        rows.append(row)
        print(k, fn, round(time.time() - t0, 2),
              "diff_d_line=", row["diff_d_line"], "diff_d_peak=", row["diff_d_peak_median"],
              "diff_P_d_line=", row["diff_P_d_line"], "diff_P_d_peak=", row["diff_P_d_peak"], flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "reproduce_diff.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    successful_files = [fn for fn in files if fn not in known_bearing_failures]
    check_extract_profiles_coverage(successful_files, problems)

    bearing_ok = out.dropna(subset=["diff_d_line", "diff_d_peak_median"])
    pin_ok = out.dropna(subset=["diff_P_d_line", "diff_P_d_peak"])
    n_total = len(out)

    print(f"\n총 {n_total}장")
    print(f"베어링 비교 가능 {len(bearing_ok)}장, max|diff_d_line|={bearing_ok['diff_d_line'].max():.3e}px, "
          f"max|diff_d_peak_median|={bearing_ok['diff_d_peak_median'].max():.3e}px")
    print(f"핀 비교 가능 {len(pin_ok)}장, max|diff_P_d_line|={pin_ok['diff_P_d_line'].max():.3e}px, "
          f"max|diff_P_d_peak|={pin_ok['diff_P_d_peak'].max():.3e}px")

    if problems:
        report_and_exit(problems)

    print(f"\n완료 기준(|Δ|<{DIFF_TOL_PX:.0e}px, 파일 집합·오류 상태 일치): PASS")


if __name__ == "__main__":
    main()
