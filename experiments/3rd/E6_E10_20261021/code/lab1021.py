"""10/21 3차 촬영 전용 측정 래퍼.

원본 shaftlab(experiments/2nd/shaftlab)은 읽기 전용으로 import만 한다.
바꾸는 것은 '사진 속 위치(1/4 좌표)'와 '물체 크기(높이 범위, 핀 상대 창)'뿐이다.
엣지 모양 상수(edge_subpx win=20, 배경 샘플 25~45px, Gaussian (1,7) σ1.2, 프로파일 ±80)는
NO2 본시험 10장 프로파일에서 밝은 테두리가 ±45px 안에 없음을 확인하고 원본 그대로 둔다(PREREG.md).
"""
import os, sys
import numpy as np, cv2

REPO2ND = os.environ.get('REPO2ND', os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '2nd')))  # experiments/2nd
sys.path.insert(0, REPO2ND)
import shaftlab.measure as SM  # noqa: E402

# ---- 10/21 설정 (1/4 축소 좌표, NO2 기준 프레임) ----
COARSE_ROWS = (700, 1080)                 # 원본 800:1200 -> 박스 창 안(위 밝음, 아래 V블록 그림자 전)에서 끝나게
BEARING = dict(x0=1975, x1=2130, hmin=177, hmax=200)   # 원본 1940~2060, 128~145 (베어링 753px/4 x 0.939~1.064)
PIN = dict(x0=2600, x1=2780, hmin=144, hmax=163)       # Ø16 기준물 611.5px/4 x 0.939~1.064
PIN_REL_WINDOW = (80, 600)               # Ø16 핀 직선부(모따기 끝 rel≈60 이후, xR까지 80px 여유). NO2 핀 데이터만 보고 결정(PREREG 3절)
NO1_OFFSET_Q = (-602, -278)               # NO1 디지털줌 크롭: 전체해상도 (-2408,-1112) / 4

_ROWS = list(COARSE_ROWS)

def coarse_section(q, x0, x1, hmin, hmax):
    """원본 coarse_section과 같은 산술. 탐색 행만 _ROWS로 바꿨다."""
    r0, r1 = _ROWS
    g = cv2.GaussianBlur(q, (3, 3), 0)
    rows = []
    for x in range(x0, x1):
        col = g[r0:r1, x].astype(int)
        dark = col < 110
        idx = np.flatnonzero(dark)
        if len(idx) == 0:
            rows.append((x, 0, 0, 0)); continue
        bright = np.flatnonzero(~dark)
        if len(bright) == 0:
            rows.append((x, 0, 0, 0)); continue
        idx = idx[idx > bright[0]]
        if len(idx) == 0:
            rows.append((x, 0, 0, 0)); continue
        t, b = idx[0], idx[-1]
        rows.append((x, b - t, t + r0, b + r0))
    rows = np.array(rows)
    ok = (rows[:, 1] >= hmin) & (rows[:, 1] <= hmax)
    best, s = (0, 0), None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s > best[1] - best[0]: best = (s, i)
            s = None
    return rows[best[0]:best[1]]

SM.coarse_section = coarse_section  # measure_bearing/measure_pin이 전역 이름으로 부른다

def set_frame(offset_q=(0, 0)):
    dx, dy = offset_q
    _ROWS[0], _ROWS[1] = COARSE_ROWS[0] + dy, COARSE_ROWS[1] + dy
    SM.BEARING_COARSE = dict(BEARING, x0=BEARING['x0'] + dx, x1=BEARING['x1'] + dx)
    SM.PIN_COARSE = dict(PIN, x0=PIN['x0'] + dx, x1=PIN['x1'] + dx)

# ---- moment_centroid (edge_methods.py 원본 복사: 그 모듈은 scipy를 import해서 이 VM에서 못 불러옴) ----
def _grad_peak_index(prof, center, win, sign):
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof); seg = g[a:b] * sign
    return int(np.argmax(seg)) + a, g

def moment_centroid(prof, center, win, sign, radius=15):
    i, g = _grad_peak_index(prof, center, win, sign)
    lo, hi = max(1, i - radius), min(len(prof) - 1, i + radius)
    idx = np.arange(lo, hi + 1); w = np.clip(g[lo:hi + 1] * sign, 0, None)
    return float(i) if w.sum() <= 0 else float((idx * w).sum() / w.sum())

def moment_diameter(gray, seg, xa, xb, step, win=45):
    """E6 scan_values.moment_diameter와 같은 산술."""
    diffs = []
    for X in range(xa, xb, step):
        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = gray[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        diffs.append(moment_centroid(prof, bc - y0, win, +1) - moment_centroid(prof, tc - y0, win, -1))
    return float(np.median(diffs)) if diffs else float('nan')

def measure_all(gray, offset_q=(0, 0), half_window_extract=44, want_profiles=True):
    """E6 scan_values 한 장분 + E7 build_cache 한 장분을 한 번의 디코딩으로."""
    set_frame(offset_q)
    q = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    seg_b = coarse_section(q, **SM.BEARING_COARSE)
    rb = SM.measure_bearing(gray, edge_method='cross50')
    xa, xb = int(seg_b[0, 0] * 4), int(seg_b[-1, 0] * 4)
    mrg = int((xb - xa) * 0.2)
    out = dict(bearing_cross50_px=rb['d_line'], bearing_grad_peak_px=rb['d_peak_median'],
               bearing_moment_px=moment_diameter(gray, seg_b, xa + mrg, xb - mrg, step=4),
               bearing_x_from=rb['x_from'], bearing_x_to=rb['x_to'], bearing_n_cols=rb['n_cols'],
               bearing_top_y=rb['top_y'], bearing_bot_y=rb['bot_y'], bearing_tilt_deg=rb['tilt_deg'],
               bearing_edge_width=rb['edge_width'], bearing_bg=rb['bg_level'], bearing_fg=rb['fg_level'],
               bearing_seg_len=len(seg_b))
    rp = SM.measure_pin(gray, edge_method='cross50', position_method='pin_relative', rel_window=PIN_REL_WINDOW)
    out.update(pin_cross50_px=rp['d_line'], pin_grad_peak_px=rp['d_peak_median'],
               pin_window_clipped=bool(rp['window_clipped']), pin_xL=rp['xL'], pin_xR=rp['xR'],
               pin_top_y=rp['top_y'])
    seg_p = coarse_section(q, **SM.PIN_COARSE)
    xL, xR = int(seg_p[0, 0] * 4), int(seg_p[-1, 0] * 4)
    pa, pb = xL + PIN_REL_WINDOW[0], min(xL + PIN_REL_WINDOW[1], xR)
    out['pin_moment_px'] = moment_diameter(gray, seg_p, pa, pb, step=2) if pb - pa >= 6 else float('nan')
    prof = None
    if want_profiles:
        prof = SM.extract_profiles(gray, rb, half_window=half_window_extract, step=2,
                                   return_labels=True, coarse_seg=seg_b)
    return out, prof
