"""E7 데이터 생성: 이미지 130장 x 열 x {상단,하단} 1D 밝기 프로파일 + teacher(cross50)
서브픽셀 라벨을 뽑아 `experiments/2nd/cache/E7_edge1d/profiles.npz`에 저장한다
(규칙 6 — cache/는 커밋하지 않는다, 재실행으로 재생성 가능).

- 베어링부 section은 `measure_bearing(gray, edge_method="cross50")`(E1, 수정 안 함)로
  한 번만 구한다. 프로파일·라벨은 `shaftlab.measure.extract_profiles(..., return_labels=True,
  coarse_seg=...)`(E7에서 추가한 옵션)로 뽑는다 — teacher = cross50(E4에서 1순위로 선정,
  `E4_edge_benchmark/README.md`).
- **창 중심을 teacher가 아니라 `coarse_section()`의 1/4 해상도 coarse 추정으로 잡는다**
  (Codex adversarial-review [high] 반영, 2차). 처음 버전은 `extract_profiles`의 기본
  동작대로 `round(teacher)`를 중심으로 썼는데, 그러면 라벨(중심→teacher 거리)이 반올림
  오차(±0.5px)뿐이라 "teacher 좌표가 모델 입력 중심에 이미 새어 들어가는" 순환 설계였다
  — 모델이 "엣지를 찾는" 게 아니라 "teacher 자신의 반올림 오차만 보정"하는 과제를 푼
  꼴이었다. 지금은 `_measure_window`가 `edge_subpx`를 부르기 **전** 쓰는 바로 그 coarse
  좌표(1/4 해상도 `coarse_section` 결과, teacher와 무관)를 중심으로 잡는다 — 실측
  오프셋 범위는 상단 -4~0px, 하단 +6~+15px(13개 셋업 샘플 기준, 설정값에 따라 바뀔 수
  있음) — `half_window=40`이 이 범위를 충분히 덮는다.
- **shift 증강이 외삽 없이 동작하도록 창을 실제로 조금 더 넓게 뽑는다**(Codex
  adversarial-review [medium] 반영) — 처음 버전은 81px 창 자체에 `CubicSpline(...,
  extrapolate=True)`로 shift를 적용했는데, ±0.5px 이동이면 창 양 끝 중 한쪽은 실제 데이터
  범위를 벗어나 스플라인이 "외삽"한 값을 써야 했다(방향마다 반대쪽 끝이 외삽돼 비대칭
  아티팩트 가능성). 지금은 `SHIFT_BUFFER_PX=4`만큼 더 넓게(`half_window+buffer`) 뽑아
  저장하고, `augment.py`의 `shift1d`가 이 여유분 안에서만 보간하도록 했다 — ±0.5px
  이동은 버퍼 안에 넉넉히 들어온다(필요한 여유는 0.5px, 버퍼는 4px).
- `half_window_px`는 지시서 규칙 14(해상도·위치 독립)를 따라 모듈 상단 상수로 뺐다 —
  다음 촬영이 50MP라도 같은 "물리적" 창 폭을 쓰려면 이 값을 1px당 mm 스케일에 맞춰
  다시 계산하면 된다(현재 스케일 1px≈0.0367mm, 0.2절).
- `col_step_px=2`는 `measure_bearing`이 `top_y_per_col`/`bot_y_per_col`을 채우는
  간격과 같다(E1 `measure_section` 기본값) — 서브샘플링 없이 셋업당 ~74~75열을 전부 쓴다
  (지시서가 추정한 "130x75x2≈1.95만 개"와 일치하는 밀도).
- NG1(묶음11)·NG2(묶음12)도 라벨은 뽑아 저장한다 — 학습에 쓸지는 `train_eval.py`가
  규칙 4(NG는 평가에만)에 따라 분할 단계에서 가른다. 여기서는 전부 저장만 한다.
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import coarse_section, paths  # noqa: E402
from shaftlab.measure import BEARING_COARSE, extract_profiles, measure_bearing  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "cache", "E7_edge1d")
CACHE_NPZ = os.path.join(CACHE_DIR, "profiles.npz")

HALF_WINDOW_PX = 40  # 규칙 14: 물리 길이 고정, 다음 촬영이 50MP면 이 상수를 스케일에 맞춰 바꾼다
SHIFT_BUFFER_PX = 4  # shift1d가 외삽 없이 보간할 여유분(필요한 건 0.5px, 넉넉히 4px)
EXTRACT_HALF_WINDOW = HALF_WINDOW_PX + SHIFT_BUFFER_PX  # 저장 시 실제로 뽑는 반창
COL_STEP_PX = 2  # measure_bearing/measure_section 기본 step과 동일 (서브샘플링 없음)

META_COLS = ["묶음", "시편", "역할", "등급", "실측mm", "초점", "핀배치", "시각"]


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = pd.read_csv(paths.EDA_MEASURE_RAW_CSV, encoding="utf-8-sig").set_index("file")
    meta = meta[meta["시편"].notna()]
    files = sorted(meta.index)

    top_profiles, bot_profiles = [], []
    top_label, bot_label = [], []
    top_center, bot_center = [], []
    xs_all, file_all = [], []
    bundle_all, specimen_all, role_all, focus_all, pin_all = [], [], [], [], []

    t_start = time.time()
    for k, fn in enumerate(files):
        t0 = time.time()
        gray = cv2.imread(os.path.join(paths.IMAGE_DIR, fn), cv2.IMREAD_GRAYSCALE)
        q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        seg = coarse_section(q, **BEARING_COARSE)
        r = measure_bearing(gray, edge_method="cross50")
        profiles, labels, xs, centers = extract_profiles(
            gray, r, half_window=EXTRACT_HALF_WINDOW, step=COL_STEP_PX,
            return_labels=True, coarse_seg=seg,
        )
        n = len(xs)
        m = meta.loc[fn]
        top_profiles.append(profiles[:, 0, :])
        bot_profiles.append(profiles[:, 1, :])
        top_label.append(labels[:, 0])
        bot_label.append(labels[:, 1])
        top_center.append(centers[:, 0])
        bot_center.append(centers[:, 1])
        xs_all.append(xs)
        file_all += [fn] * n
        bundle_all += [int(m["묶음"])] * n
        specimen_all += [str(m["시편"])] * n
        role_all += [str(m["역할"])] * n
        focus_all += [str(m["초점"])] * n
        pin_all += [str(m["핀배치"])] * n
        print(k, fn, n, round(time.time() - t0, 2), flush=True)

    out = dict(
        top_profiles=np.concatenate(top_profiles, axis=0).astype(np.float32),
        bot_profiles=np.concatenate(bot_profiles, axis=0).astype(np.float32),
        top_label=np.concatenate(top_label, axis=0).astype(np.float32),
        bot_label=np.concatenate(bot_label, axis=0).astype(np.float32),
        top_center=np.concatenate(top_center, axis=0).astype(np.int32),
        bot_center=np.concatenate(bot_center, axis=0).astype(np.int32),
        x=np.concatenate(xs_all, axis=0).astype(np.int32),
        filename=np.array(file_all),
        bundle=np.array(bundle_all, dtype=np.int32),
        specimen=np.array(specimen_all),
        role=np.array(role_all),
        focus=np.array(focus_all),
        pin=np.array(pin_all),
        half_window_px=np.array([HALF_WINDOW_PX]),
        shift_buffer_px=np.array([SHIFT_BUFFER_PX]),
        col_step_px=np.array([COL_STEP_PX]),
    )
    np.savez(CACHE_NPZ, **out)
    print()
    print(f"행수: {len(out['filename'])} (이미지 {len(files)}장, {round(time.time()-t_start, 1)}초)")
    print(f"프로파일 폭: {2*EXTRACT_HALF_WINDOW+1} (모델 입력 81 + shift 버퍼 {SHIFT_BUFFER_PX}px 양쪽)")
    print(f"저장: {CACHE_NPZ}")


if __name__ == "__main__":
    main()
