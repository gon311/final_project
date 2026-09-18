"""
================================================================
 캘리브레이션만 다시 잡아 측정 지표를 재계산 — 학습 없음
================================================================
 하는 일 : 이미 학습된 best.pt로 새 calib 이미지 3~5장만 추론해서
           mm_per_pixel을 다시 구하고, per_image CSV에 이미 저장된
           measured_pixel로 bias / MAE / RMSE / σ를 다시 계산한다.
 왜 필요 : 측정값(px)은 캘리브레이션과 무관하게 이미 구해져 있다.
           스케일 상수만 틀렸을 때 100 epoch를 다시 돌릴 이유가 없다.
 사용법  : 아래 [설정] 3줄만 맞추고 실행. 원본 결과 파일은 건드리지 않고
           *_recalib 이름으로 따로 저장한다.
 ★ 주의  : run_my_model.py 를 재실행하면 last.pt 때문에 resume 경로로 들어간다.
           학습이 이미 끝난 run은 resume이 되지 않으므로 이 파일을 쓴다.
"""

# ══════════ [설정] 여기 3줄만 ══════════
RUN_NAME            = "yolo26n"    # runs/ 아래 내 폴더 이름 (MODEL에서 .pt 뗀 것)
NEW_CALIBRATION_MM  = 20.026       # ★ 새 calib 시편의 실측 외경. specimen_map.csv 의 true_mm 을 그대로
DRIVE               = "/content/drive/MyDrive/shaft_sweep"
# ══════════════════════════════════════

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

root = Path(DRIVE)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import run_my_model as R
except ImportError:
    sys.exit("[중단] run_my_model.py 를 같은 폴더에 두고 실행하세요.")

from ultralytics import YOLO

# ── 1) 학습된 가중치 ─────────────────────────────────────────
weights = root / "runs" / RUN_NAME / "weights" / "best.pt"
if not weights.exists():
    sys.exit(f"[중단] 학습된 가중치가 없습니다: {weights}\n  RUN_NAME 을 확인하세요.")
model = YOLO(str(weights))
print(f"[1/4] 가중치 로드: {weights}")

# ── 2) 새 calib 이미지로 mm_per_pixel 재산출 ─────────────────
calib_dir = root / "dataset" / "calib"
calib_images = R.image_files(calib_dir)
if not calib_images:
    sys.exit(f"[중단] calib 이미지가 없습니다: {calib_dir}")

rows = [{"image": p.name, **R.measure(p, model, None, cv2, np)} for p in calib_images]
pixels = [r["measured_pixel"] for r in rows if r["status"] == "ok"]
for r in rows:
    print(f"   {r['image']:<32} {r['status']:<20} "
          f"{r['measured_pixel'] if r['measured_pixel'] is None else round(r['measured_pixel'], 3)}")

if len(pixels) < R.MIN_CALIB_IMAGES:
    sys.exit(f"[중단] 유효한 calib 이미지가 {len(pixels)}장뿐입니다 (최소 {R.MIN_CALIB_IMAGES}장).")
arr = np.array(pixels, dtype=float)
calibration_pixel = float(np.median(arr))
spread_pct = float((arr.max() - arr.min()) / calibration_pixel * 100.0)
if spread_pct > R.CALIB_SPREAD_MAX_PCT:
    sys.exit(f"[중단] calib 이미지끼리 픽셀이 너무 벌어집니다: "
             f"{arr.min():.2f}~{arr.max():.2f}px (폭 {spread_pct:.1f}%). 같은 시편·같은 세션인지 확인하세요.")
mm_per_pixel = float(NEW_CALIBRATION_MM / calibration_pixel)
print(f"[2/4] 새 캘리브레이션: {len(pixels)}장, 중앙값 {calibration_pixel:.3f}px "
      f"(폭 {spread_pct:.2f}%), mm_per_pixel={mm_per_pixel:.6f}")

# ── 3) 기존 per_image CSV의 measured_pixel 재사용 ────────────
per_image_path = root / "results" / f"per_image_{RUN_NAME}.csv"
if not per_image_path.exists():
    sys.exit(f"[중단] {per_image_path} 가 없습니다. run_my_model.py 를 한 번은 완주해야 합니다.")
df = pd.read_csv(per_image_path)
calib_names = {p.name for p in calib_images}
df = df[~df["image"].isin(calib_names)].copy()          # calib 이미지는 오차 평가에서 제외
ok = df[df["status"] == "ok"].copy()
if ok.empty:
    sys.exit("[중단] per_image CSV에 status=ok 행이 없습니다.")

gap_pct = abs(calibration_pixel - float(ok["measured_pixel"].median())) / float(ok["measured_pixel"].median()) * 100.0
if gap_pct > R.CALIB_SCALE_MAX_PCT:
    sys.exit(f"[중단] calib 배율이 측정 이미지와 {gap_pct:.1f}% 다릅니다 (허용 {R.CALIB_SCALE_MAX_PCT:.0f}%).\n"
             f"  calib {calibration_pixel:.2f}px vs 본 이미지 중앙값 {ok['measured_pixel'].median():.2f}px\n"
             "  → calib 이미지가 161장과 다른 촬영 세션입니다. 교체하세요.")
print(f"[3/4] 배율 점검 OK: calib {calibration_pixel:.2f}px vs 본 이미지 "
      f"{ok['measured_pixel'].median():.2f}px (차이 {gap_pct:.2f}%)")

old_bias_um = float((ok["measured_mm"] - ok["true_mm"]).mean() * 1000.0)
ok["measured_mm"] = ok["measured_pixel"] * mm_per_pixel
ok["error_mm"] = ok["measured_mm"] - ok["true_mm"]
ok["predicted_ng"] = ~ok["measured_mm"].between(R.SPEC_LOWER, R.SPEC_UPPER)

err = ok["error_mm"].to_numpy(dtype=float)
sigma = float(ok.groupby("specimen_id")["measured_mm"].std(ddof=0).mean() * 1000.0)
spec = (ok.groupby("specimen_id", as_index=False)
          .agg(measured_mm=("measured_mm", "median"),
               true_mm=("true_mm", "first"),
               is_ng=("is_ng", "first"),
               px=("measured_pixel", "median"),
               n=("image", "count")))
spec["predicted_ng"] = ~spec["measured_mm"].between(R.SPEC_LOWER, R.SPEC_UPPER)
spec["error_um"] = (spec["measured_mm"] - spec["true_mm"]) * 1000.0

metrics = {
    "calibration_mm": NEW_CALIBRATION_MM,
    "calibration_pixel": calibration_pixel,
    "mm_per_pixel": mm_per_pixel,
    "calib_n_used": len(pixels),
    "calib_spread_pct": spread_pct,
    "scale_gap_pct": gap_pct,
    "bias_um": float(err.mean() * 1000.0),
    "mae_um": float(np.abs(err).mean() * 1000.0),
    "rmse_um": float(np.sqrt(np.square(err).mean()) * 1000.0),
    "repeat_sigma_um": sigma,
    "ng_detected": int((spec["is_ng"] & spec["predicted_ng"]).sum()),
    "false_alarm": int((~spec["is_ng"] & spec["predicted_ng"]).sum()),
    "n_ng_specimens": int(spec["is_ng"].sum()),
    "n_eval_images": int(len(ok)),
    "n_eval_specimens": int(spec["specimen_id"].nunique()),
}

# ── 4) 저장 + 요약 (원본 파일은 덮어쓰지 않는다) ─────────────
out_csv = root / "results" / f"per_image_{RUN_NAME}_recalib.csv"
out_json = root / "results" / f"result_{RUN_NAME}_recalib.json"
ok.to_csv(out_csv, index=False, encoding="utf-8-sig")
out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("\n========== 시편별 ==========")
print(f"{'시편':<8}{'n':>4}{'px':>10}{'측정mm':>11}{'실측mm':>10}{'오차µm':>10}  {'NG?':<6}{'판정'}")
for r in spec.sort_values("specimen_id").itertuples(index=False):
    print(f"{str(r.specimen_id):<8}{r.n:>4}{r.px:>10.2f}{r.measured_mm:>11.4f}"
          f"{r.true_mm:>10.3f}{r.error_um:>10.0f}  {'NG' if r.is_ng else '정상':<6}"
          f"{'규격밖' if r.predicted_ng else '규격안'}")

print("\n========== 재계산 결과 ==========")
print(f"1. 캘리브레이션 : {calibration_pixel:.3f}px / {NEW_CALIBRATION_MM}mm "
      f"= {mm_per_pixel:.6f} mm/px  ({len(pixels)}장, 폭 {spread_pct:.2f}%)")
print(f"2. 배율 점검    : 본 이미지와 차이 {gap_pct:.2f}% (허용 {R.CALIB_SCALE_MAX_PCT:.0f}%)")
print(f"3. 측정 정확도  : bias={metrics['bias_um']:.1f}µm, MAE={metrics['mae_um']:.1f}µm, "
      f"RMSE={metrics['rmse_um']:.1f}µm   (이전 bias {old_bias_um:.1f}µm)")
print(f"4. 측정 재현성  : σ={metrics['repeat_sigma_um']:.3f}µm")
print(f"5. 불량 검출    : NG {metrics['ng_detected']}/{metrics['n_ng_specimens']} 검출, "
      f"정상 오탐 {metrics['false_alarm']}건")
print(f"6. 저장 완료    : {out_json}")
print(f"                 {out_csv}")
print("\n※ 원본 result_/per_image_ 파일은 그대로 두었습니다. 비교용으로 남겨두세요.")
