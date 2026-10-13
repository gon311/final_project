"""
================================================================
 ROI / 박스 비교 — 같은 이미지에 여러 가중치의 결과를 겹쳐 본다
================================================================
 하는 일 : 한 장의 이미지에 대해 가중치별로
           (1) YOLO 박스, (2) 세로 40% 확장 ROI, (3) 31개 측정선의 엣지 점
           을 색을 달리해 겹쳐 그리고, 숫자 진단표를 출력한다.
 왜 필요 : "선이 박스 밖으로 나간다"는 현상이 가중치(박스) 탓인지
           확인하려면 같은 좌표계 위에 겹쳐 봐야 한다.
 사용법  : python compare_roi.py IMAGE.jpg old=/path/best.pt new=/path/best.pt
           (라벨 비교) --label /path/IMAGE.txt
 측정 로직은 measurement_engine.py V1과 동일한 파라미터를 쓴다.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── V1과 동일한 측정 규격 ───────────────────────────────────
CONF, IMGSZ = 0.25, 640
ROI_VERTICAL_MARGIN = 0.40
X_START, X_END, N_LINES = 0.18, 0.82, 31
UPPER_END, LOWER_START = 0.46, 0.54
MIN_DIAMETER, MAX_DIAMETER = 0.25, 0.60
GRAD_PERCENTILE, STRENGTH_RATIO = 82, 0.55
MAD_K = 3.0

# BGR — 가중치 순서대로 사용
COLORS = [(0, 0, 255), (0, 200, 0), (255, 140, 0), (255, 0, 255)]
LABEL_COLOR = (0, 255, 255)


def subpixel_edge(abs_gradient, index):
    if index <= 0 or index >= len(abs_gradient) - 1:
        return float(index)
    g1, g2, g3 = map(float, abs_gradient[index - 1:index + 2])
    denominator = g1 - 2.0 * g2 + g3
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index) + float(np.clip(0.5 * (g1 - g3) / denominator, -1.0, 1.0))


def find_outer_edge(gray_image, x):
    strip_x1, strip_x2 = max(0, x - 2), min(gray_image.shape[1], x + 3)
    profile = np.mean(gray_image[:, strip_x1:strip_x2], axis=1).astype(np.float32)
    gradient = np.gradient(profile)
    abs_gradient = np.abs(gradient)
    h = len(profile)
    upper_end, lower_start = int(h * UPPER_END), int(h * LOWER_START)
    upper_abs, lower_abs = abs_gradient[:upper_end], abs_gradient[lower_start:]
    if len(upper_abs) == 0 or len(lower_abs) == 0:
        return None
    upper_candidates = np.where(upper_abs >= np.percentile(upper_abs, GRAD_PERCENTILE))[0]
    lower_candidates = np.where(lower_abs >= np.percentile(lower_abs, GRAD_PERCENTILE))[0] + lower_start
    pairs = []
    for top_idx in upper_candidates:
        top_gradient = float(gradient[top_idx])
        for bottom_idx in lower_candidates:
            bottom_gradient = float(gradient[bottom_idx])
            if top_gradient * bottom_gradient >= 0:
                continue
            diameter = bottom_idx - top_idx
            if diameter < h * MIN_DIAMETER or diameter > h * MAX_DIAMETER:
                continue
            pairs.append((int(top_idx), int(bottom_idx), float(diameter),
                          abs(top_gradient) + abs(bottom_gradient)))
    if not pairs:
        return None
    limit = max(pair[3] for pair in pairs) * STRENGTH_RATIO
    strong = [pair for pair in pairs if pair[3] >= limit] or pairs
    top_idx, bottom_idx = map(int, max(strong, key=lambda pair: pair[2])[:2])
    top_sub = subpixel_edge(abs_gradient, top_idx)
    bottom_sub = subpixel_edge(abs_gradient, bottom_idx)
    return top_sub, bottom_sub, bottom_sub - top_sub


def measure_with_box(image, box):
    """박스(x1,y1,x2,y2)를 받아 V1과 같은 절차로 측정하고 진단값을 함께 돌려준다."""
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    margin = int(box_h * ROI_VERTICAL_MARGIN)
    measure_y1, measure_y2 = max(0, y1 - margin), min(image_h, y2 + margin)
    roi = image[measure_y1:measure_y2, x1:x2]
    if roi.size == 0:
        return None
    blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi_h, roi_w = blur.shape
    x_positions = np.linspace(int(roi_w * X_START), int(roi_w * X_END), N_LINES).astype(int)
    points = []          # (원본 x, 원본 top_y, 원본 bottom_y, 지름)
    for x in x_positions:
        edge = find_outer_edge(blur, x)
        if edge is None:
            continue
        top_sub, bottom_sub, diameter = edge
        points.append((x1 + x, measure_y1 + top_sub, measure_y1 + bottom_sub, diameter))
    if not points:
        return None
    diameters = np.array([p[3] for p in points], dtype=float)
    deviation = np.abs(diameters - np.median(diameters))
    mad = float(np.median(deviation))
    valid = deviation <= MAD_K * 1.4826 * mad if mad > 0 else np.ones(len(diameters), bool)
    median_diameter = float(np.median(diameters[valid])) if valid.any() else float(np.median(diameters))
    # 박스 밖으로 나간 라인 수 — "선이 박스 밖으로 뻗는다"의 정량화
    outside = sum(1 for p in points if p[1] < y1 - 0.5 or p[2] > y2 + 0.5)
    return {
        "box": (x1, y1, x2, y2), "box_h": box_h,
        "roi": (x1, measure_y1, x2, measure_y2), "roi_h": roi_h,
        "points": points, "valid": valid,
        "n_measured": len(points), "n_valid": int(valid.sum()),
        "n_outside_box": outside,
        "median_px": median_diameter,
        "std_px": float(np.std(diameters[valid])) if valid.any() else float("nan"),
        "ratio_roi": median_diameter / roi_h,      # 0.60 상한에 얼마나 붙었나
        "ratio_box": median_diameter / box_h,      # 1.0에 가까울수록 박스가 외곽선에 맞음
    }


def detect_box(weights_path, image_path, image_shape):
    model = YOLO(str(weights_path))
    predictions = model.predict(source=str(image_path), conf=CONF, imgsz=IMGSZ, verbose=False)
    candidates = [(float(b.conf[0]), b) for p in predictions if p.boxes is not None for b in p.boxes]
    if not candidates:
        return None, None
    conf, best = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
    image_h, image_w = image_shape[:2]
    return (max(0, int(x1)), max(0, int(y1)), min(image_w, int(x2)), min(image_h, int(y2))), conf


def read_label_box(label_path, image_shape):
    """YOLO txt(정규화 xywh) → 픽셀 xyxy."""
    image_h, image_w = image_shape[:2]
    lines = [line for line in Path(label_path).read_text().splitlines() if line.strip()]
    if not lines:
        return None
    _, xc, yc, bw, bh = (float(v) for v in lines[0].split())
    x1 = int(round((xc - bw / 2) * image_w))
    y1 = int(round((yc - bh / 2) * image_h))
    x2 = int(round((xc + bw / 2) * image_w))
    y2 = int(round((yc + bh / 2) * image_h))
    return max(0, x1), max(0, y1), min(image_w, x2), min(image_h, y2)


def draw(canvas, result, color, name):
    x1, y1, x2, y2 = result["box"]
    rx1, ry1, rx2, ry2 = result["roi"]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)                 # YOLO 박스
    for x in range(rx1, rx2, 24):                                        # 확장 ROI(점선)
        cv2.line(canvas, (x, ry1), (min(x + 12, rx2), ry1), color, 1)
        cv2.line(canvas, (x, ry2), (min(x + 12, rx2), ry2), color, 1)
    for (px, top_y, bottom_y, _), ok in zip(result["points"], result["valid"]):
        point_color = color if ok else (0, 0, 255)
        cv2.line(canvas, (int(px), int(round(top_y))), (int(px), int(round(bottom_y))), point_color, 1)
        cv2.circle(canvas, (int(px), int(round(top_y))), 3, point_color, -1)
        cv2.circle(canvas, (int(px), int(round(bottom_y))), 3, point_color, -1)
    cv2.putText(canvas, name, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    parser = argparse.ArgumentParser(description="가중치별 YOLO 박스·ROI·측정선 비교")
    parser.add_argument("image", help="비교할 이미지 (팀 dataset의 2992px 버전 권장)")
    parser.add_argument("weights", nargs="+", help="이름=경로 형식. 예: old=./best.pt new=./runs/yolo26n/weights/best.pt")
    parser.add_argument("--label", help="검수 라벨 txt (있으면 노란색으로 함께 표시)")
    parser.add_argument("--out", default="compare_roi.jpg", help="출력 이미지 경로")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"[중단] 이미지를 읽을 수 없습니다: {args.image}")
    print(f"이미지: {args.image}  (가로 {image.shape[1]}px, 세로 {image.shape[0]}px)")

    canvas = image.copy()
    rows, boxes_for_crop = [], []

    for index, spec in enumerate(args.weights):
        name, _, weights_path = spec.partition("=")
        if not weights_path:
            raise SystemExit(f"[중단] '이름=경로' 형식이어야 합니다: {spec}")
        box, conf = detect_box(weights_path, args.image, image.shape)
        if box is None:
            print(f"[{name}] 검출 실패")
            continue
        result = measure_with_box(image, box)
        if result is None:
            print(f"[{name}] 박스는 나왔지만 측정 실패 box={box}")
            continue
        color = COLORS[index % len(COLORS)]
        draw(canvas, result, color, name)
        boxes_for_crop.append(result["roi"])
        rows.append((name, conf, result))

    if args.label:
        label_box = read_label_box(args.label, image.shape)
        if label_box:
            lx1, ly1, lx2, ly2 = label_box
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), LABEL_COLOR, 2)
            cv2.putText(canvas, "label", (lx1, min(image.shape[0] - 10, ly2 + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_COLOR, 2)
            boxes_for_crop.append(label_box)
            print(f"검수 라벨 박스: y1={ly1} y2={ly2} 높이={ly2 - ly1}")

    if not rows:
        raise SystemExit("[중단] 비교할 결과가 없습니다.")

    print()
    print(f"{'이름':<10}{'conf':>7}{'box_y1':>8}{'box_y2':>8}{'box_h':>7}"
          f"{'측정px':>10}{'std':>8}{'D/ROI':>8}{'D/box':>8}{'유효':>7}{'박스밖':>8}")
    for name, conf, r in rows:
        x1, y1, x2, y2 = r["box"]
        print(f"{name:<10}{conf:>7.3f}{y1:>8}{y2:>8}{r['box_h']:>7}"
              f"{r['median_px']:>10.2f}{r['std_px']:>8.2f}{r['ratio_roi']:>8.3f}"
              f"{r['ratio_box']:>8.3f}{r['n_valid']:>4}/{r['n_measured']:<2}{r['n_outside_box']:>8}")
    print()
    print("해석 기준")
    print(f"  D/ROI  : 측정 지름 ÷ 확장 ROI 높이. 상한 {MAX_DIAMETER} 에 가까우면 진짜 외곽이 탈락하기 직전이다.")
    print("  D/box  : 측정 지름 ÷ YOLO 박스 높이. 1.0 근처면 박스가 외곽선에 맞은 것, 1.1 이상이면 박스가 작다.")
    print("  박스밖 : 31개 측정선 중 엣지가 YOLO 박스 밖에서 잡힌 라인 수. 클수록 배경 엣지를 물었을 가능성이 크다.")

    cv2.imwrite(args.out, canvas)
    crop_x1 = max(0, min(b[0] for b in boxes_for_crop) - 60)
    crop_y1 = max(0, min(b[1] for b in boxes_for_crop) - 60)
    crop_x2 = min(image.shape[1], max(b[2] for b in boxes_for_crop) + 60)
    crop_y2 = min(image.shape[0], max(b[3] for b in boxes_for_crop) + 60)
    crop_path = str(Path(args.out).with_name(Path(args.out).stem + "_zoom.jpg"))
    cv2.imwrite(crop_path, canvas[crop_y1:crop_y2, crop_x1:crop_x2])
    print(f"\n저장: {args.out}\n      {crop_path} (확대본 — 이걸 보세요)")


if __name__ == "__main__":
    main()
