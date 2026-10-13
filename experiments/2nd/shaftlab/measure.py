"""서브픽셀 외경(px) 측정 모듈.

`DATA_ROOT/EDA_20261008/scripts/eda_measure.py`(읽기 전용, 수정하지 않음)의
`coarse_section`/`edge_subpx`/`measure_section`을 그대로 옮기고(E1, 0.4절),
그 위에 import 가능한 공개 인터페이스(`measure_bearing`, `measure_pin`,
`extract_profiles`)를 얹었다. 원본과 동일한 산술을 유지해야 기존
`eda_measure_raw.csv`/`pin_gauge_raw.csv`가 그대로 재현된다.
"""
from __future__ import annotations

import cv2
import numpy as np

EdgeMethod = str  # "cross50" | "grad_peak"

# 베어링부/핀 1/4 축소 영상 탐색 구간 (eda_measure.py, eda_pin_gauge.py와 동일)
BEARING_COARSE = dict(x0=1940, x1=2060, hmin=128, hmax=145)
PIN_COARSE = dict(x0=2385, x1=2450, hmin=58, hmax=72)


def coarse_section(q, x0, x1, hmin, hmax):
    """1/4 축소 영상에서 외곽 실루엣 높이 프로파일 -> 구간 [xa, xb]과 top/bottom(1/4 좌표).

    원본 그대로(eda_measure.py). 수정하지 않음.
    """
    g = cv2.GaussianBlur(q, (3, 3), 0)
    rows = []
    for x in range(x0, x1):
        col = g[800:1200, x].astype(int)
        dark = col < 110
        idx = np.flatnonzero(dark)
        if len(idx) == 0:
            rows.append((x, 0, 0, 0))
            continue
        # 상단 프레임(800 부근 어두움) 제거: 위쪽 밝은 간극 이후 첫 dark
        bright = np.flatnonzero(~dark)
        if len(bright) == 0:
            rows.append((x, 0, 0, 0))
            continue
        first_bright = bright[0]
        idx = idx[idx > first_bright]
        if len(idx) == 0:
            rows.append((x, 0, 0, 0))
            continue
        t, b = idx[0], idx[-1]
        rows.append((x, b - t, t + 800, b + 800))
    rows = np.array(rows)
    ok = (rows[:, 1] >= hmin) & (rows[:, 1] <= hmax)
    # 가장 긴 연속 구간
    best, s = (0, 0), None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s > best[1] - best[0]:
                best = (s, i)
            s = None
    seg = rows[best[0]:best[1]]
    return seg


def edge_subpx(prof, center, win, sign):
    """sign=-1: bright->dark(상단), +1: dark->bright(하단).

    gradient peak + 50% level crossing. 원본 그대로(eda_measure.py). 수정하지 않음.
    """
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof)
    seg = g[a:b] * sign
    i = int(np.argmax(seg)) + a
    g1, g2, g3 = g[i - 1] * sign, g[i] * sign, g[i + 1] * sign
    den = g1 - 2 * g2 + g3
    off = 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (g1 - g3) / den, -1, 1))
    peak = i + off
    # 50% crossing: 배경/물체 레벨은 엣지에서 25~45px 떨어진 곳의 중앙값
    if sign < 0:
        bg = np.median(prof[max(0, i - 45):max(1, i - 25)])
        fg = np.median(prof[i + 25:i + 45])
    else:
        fg = np.median(prof[max(0, i - 45):max(1, i - 25)])
        bg = np.median(prof[i + 25:i + 45])
    lvl = 0.5 * (bg + fg)
    j = i
    if sign < 0:
        while j > a and prof[j] < lvl:
            j -= 1
        while j < b and prof[j + 1] >= lvl:
            j += 1
    else:
        while j > a and prof[j] > lvl:
            j -= 1
        while j < b and prof[j + 1] <= lvl:
            j += 1
    p0, p1 = prof[j], prof[j + 1]
    cross = j + (0 if p1 == p0 else (lvl - p0) / (p1 - p0))
    # 10-90% 엣지 폭(블러 지표)
    lo, hi = fg + 0.1 * (bg - fg), fg + 0.9 * (bg - fg)
    r = prof[max(0, i - 30):i + 30]
    if sign < 0:
        r = r[::-1]
    try:
        k10 = np.flatnonzero(r >= lo)[0]
        k90 = np.flatnonzero(r >= hi)[0]
        ew = float(k90 - k10)
    except IndexError:
        ew = np.nan
    return peak, cross, float(bg), float(fg), ew, float(g[i] * sign)


def _measure_window(full, seg, xa, xb, step=2):
    """[xa, xb)(절대 x좌표) 구간만 써서 열별 측정 + 강건 직선적합을 한다.
    `measure_section`의 핵심 루프를 margin 적용 이전 단계로 뽑아낸 것뿐이라 산술은
    원본(eda_measure.py)과 똑같다 — `measure_section`은 이 함수에 margin을 적용한
    xa,xb를 넘겨 호출하도록만 바뀌었고, `measure_pin`이 핀 상대 구간처럼 xa,xb를
    직접 고를 때도 같은 함수를 쓴다(E5, 중복 구현 방지)."""
    tops, bots, xs, ews, bgs, fgs, grs = [], [], [], [], [], [], []
    tp, bp = [], []
    for X in range(xa, xb, step):
        si = np.argmin(np.abs(seg[:, 0] * 4 - X))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = full[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        t = edge_subpx(prof, tc - y0, 20, -1)
        b = edge_subpx(prof, bc - y0, 20, +1)
        xs.append(X)
        tops.append(t[1] + y0)
        bots.append(b[1] + y0)
        tp.append(t[0] + y0)
        bp.append(b[0] + y0)
        ews += [t[4], b[4]]
        bgs += [t[2], b[2]]
        fgs += [t[3], b[3]]
        grs += [t[5], b[5]]
    xs, tops, bots, tp, bp = map(np.array, (xs, tops, bots, tp, bp))

    def robust_fit(x, y):
        p = np.polyfit(x, y, 1)
        r = y - np.polyval(p, x)
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        k = np.abs(r) < 4 * 1.4826 * mad
        return np.polyfit(x[k], y[k], 1), r

    pt, rt = robust_fit(xs, tops)
    pb, rb = robust_fit(xs, bots)
    xc = xs.mean()
    slope = 0.5 * (pt[0] + pb[0])
    d_line = (np.polyval(pb, xc) - np.polyval(pt, xc)) * np.cos(np.arctan(slope))
    d_cols = bots - tops
    d_peak = np.median(bp - tp)
    return dict(
        x_from=int(xa), x_to=int(xb), n_cols=len(xs),
        d_cross_median=float(np.median(d_cols)), d_cross_std=float(np.std(d_cols)),
        d_line=float(d_line), d_peak_median=float(d_peak),
        top_y=float(np.polyval(pt, xc)), bot_y=float(np.polyval(pb, xc)),
        tilt_deg=float(np.degrees(np.arctan(slope))),
        taper_px_per_100=float((pb[0] - pt[0]) * 100),
        edge_width=float(np.nanmedian(ews)), bg_level=float(np.median(bgs)),
        fg_level=float(np.median(fgs)), grad_peak=float(np.median(grs)),
        top_resid_std=float(np.std(rt)), bot_resid_std=float(np.std(rb)),
        top_y_per_col=tops, bot_y_per_col=bots, x_per_col=xs,
    )


def measure_section(full, seg, margin_frac=0.2, step=2):
    """열별 측정 + 강건 직선적합. 원본 그대로(eda_measure.py)와 동일한 산술.
    수정하지 않음 — margin 적용 후 `_measure_window`를 호출하도록만 리팩토링했다."""
    xa, xb = seg[0, 0] * 4, seg[-1, 0] * 4
    m = int((xb - xa) * margin_frac)
    xa, xb = xa + m, xb - m
    return _measure_window(full, seg, xa, xb, step)


def _diameter_for_method(section_result: dict, edge_method: EdgeMethod) -> float:
    if edge_method == "cross50":
        return section_result["d_line"]
    if edge_method == "grad_peak":
        return section_result["d_peak_median"]
    raise ValueError(f"unknown edge_method: {edge_method!r} (cross50 | grad_peak)")


def measure_bearing(gray_full: np.ndarray, edge_method: EdgeMethod = "cross50") -> dict:
    """베어링부(약 Ø20) 서브픽셀 외경을 잰다.

    `eda_measure.py`의 베어링부 호출(1/4 축소 탐색 구간 1940~2060px, 높이 128~145px,
    margin_frac=0.2, step=2)과 동일한 산술을 쓴다. `edge_method`는 어느 엣지 정의를
    `diameter_px`로 쓸지만 바꾼다("cross50" -> d_line, "grad_peak" -> d_peak_median);
    두 값 모두 반환 dict에 들어있다.
    """
    q = cv2.resize(gray_full, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    seg = coarse_section(q, **BEARING_COARSE)
    if len(seg) < 15:
        raise ValueError(f"bearing seg too short {len(seg)}")
    r = measure_section(gray_full, seg)
    r["edge_method"] = edge_method
    r["diameter_px"] = _diameter_for_method(r, edge_method)
    return r


# 핀 측정 위치 세 방식의 창 위치 상수 (E5, `E5_pin_gauge/step2_position_compare.py`에서 비교)
PIN_FIXED_COLS = (9655, 9700)  # (b) 이미지 고정 열
PIN_RELATIVE_WINDOW = (60, 105)  # (c) 핀 왼쪽 끝 xL 기준 s = x - xL 구간. E5에서 기본값으로 선정.


def measure_pin(
    gray_full: np.ndarray,
    edge_method: EdgeMethod = "cross50",
    position_method: str = "pin_relative",
    margin_frac: float = 0.15,
    rel_window: tuple = PIN_RELATIVE_WINDOW,
    fixed_cols: tuple = PIN_FIXED_COLS,
) -> dict:
    """Ø10 핀게이지 서브픽셀 지름을 잰다.

    **기본값이 바뀌었다(E5):** 처음 구현(0.4절 재현 당시)에는 이 함수에 `position_method`가
    없었고 동작은 지금의 `"auto"`와 같았다. E5에서 측정 위치 세 방식을 비교해 `"pin_relative"`
    가 더 안정적이라고 판단했고(아래 "pin_relative" 설명, `E5_pin_gauge/README.md` 2단계),
    그 방식을 이 함수의 **기본값**으로 바꿨다. 그래서 `measure_pin(gray)`를 인자 없이 부르면
    지금은 `"auto"`가 아니라 `"pin_relative"` 결과가 나온다 — 기존 호출부가 암묵적으로
    `"auto"`를 기대하고 있었다면 값이 달라진다(Codex 3차 adversarial-review 반영).
    **어느 쪽을 쓰는지 기본값에 기대지 말고 다음 두 명시적 함수로 부르는 걸 권장한다:**
    `measure_pin_auto(gray_full, **kwargs)`(옛 기본값과 동일) /
    `measure_pin_relative(gray_full, **kwargs)`(새 기본값과 동일, 지금의 `measure_pin()`
    기본 호출과 같음). `E1_module/reproduce.py`는 `pin_gauge_raw.csv`(옛 `"auto"` 기준으로
    만들어짐)와 비교할 때 `measure_pin_auto()`를 쓴다.

    `position_method`는 측정에 쓸 열 구간(창)을 고른다 — 추정 방식(강건 직선적합)은
    세 경우 모두 같다(E5, `_measure_window`).
      - "auto": `eda_pin_gauge.py`와 동일한 자동 구간(`coarse_section` 결과에서
        `margin_frac`만큼 양끝을 자른 구간). E1이 재현한 `pin_gauge_raw.csv`와
        비교할 때는 반드시 이 값을 명시해서 써야 한다(또는 `measure_pin_auto()`를 쓴다).
      - "fixed_cols": 이미지 고정 열 `fixed_cols`(기본 9655~9700). 카메라가 밀리면
        핀의 다른 부위를 재게 된다(0.4절 (b)).
      - "pin_relative"(기본값): 핀 왼쪽 끝 xL = `coarse_section` 결과의 시작열 기준
        상대 구간 `xL + rel_window`(기본 s=60~105). E5에서 세 방식을 비교해
        기본값으로 선정했다 — 이유와 수치는 `E5_pin_gauge/README.md`,
        `results/pin_position_compare.csv` 참고. 기본값이 "auto"에서 이것으로
        바뀌었으므로, `pin_gauge_raw.csv` 기준 E1 재현 결과와 다른 값이 나온다
        (의도된 차이, E1 README에 기록).
    `xL`, `xR`(핀 왼쪽/오른쪽 끝, `coarse_section`이 찾은 전체 구간)도 반환 dict에
    들어있다. `window_clipped`는 "pin_relative" 창(`xL+rel_window`)이 실제 검출된
    핀 범위(`xR`)를 넘어가 잘렸는지를 알려준다(Codex 2차 adversarial-review 반영,
    `pin_len`이 짧게 잡힌 프레임에서 창이 V블록 그림자 쪽으로 넘어가는 걸 막는다). 이 플래그가
    True인 이미지는 하류(E5 3·5·6단계)에서 평균 계산에선 쓰되 민감도 분석으로 따로 표시한다
    (`E5_pin_gauge/README.md` "window_clipped 전파" 참고).
    """
    q = cv2.resize(gray_full, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    seg = coarse_section(q, **PIN_COARSE)
    if len(seg) < 6:
        raise ValueError(f"pin seg too short {len(seg)}")
    xL, xR = int(seg[0, 0] * 4), int(seg[-1, 0] * 4)
    window_clipped = False

    if position_method == "auto":
        m = int((xR - xL) * margin_frac)
        xa, xb = xL + m, xR - m
    elif position_method == "fixed_cols":
        xa, xb = fixed_cols
    elif position_method == "pin_relative":
        xa = xL + rel_window[0]
        xb_requested = xL + rel_window[1]
        xb = min(xb_requested, xR)
        window_clipped = xb < xb_requested
    else:
        raise ValueError(f"unknown position_method: {position_method!r} (auto | fixed_cols | pin_relative)")

    if xb - xa < 3 * 2:  # step=2일 때 최소 3열은 있어야 강건 직선적합이 된다
        raise ValueError(f"pin window too narrow after clipping: [{xa}, {xb}) (xL={xL}, xR={xR})")

    r = _measure_window(gray_full, seg, xa, xb, step=2)
    r["edge_method"] = edge_method
    r["position_method"] = position_method
    r["xL"] = xL
    r["xR"] = xR
    r["window_clipped"] = window_clipped
    r["diameter_px"] = _diameter_for_method(r, edge_method)
    return r


def measure_pin_auto(gray_full: np.ndarray, **kwargs) -> dict:
    """`measure_pin(gray_full, position_method="auto", ...)`의 명시적 별칭.
    `measure_pin()`의 기본값이 "pin_relative"로 바뀐 뒤(E5), 기본값에 기대지 않고
    "자동 구간"을 쓰고 싶은 호출부가 이름만 보고 바로 알 수 있게 뒀다
    (Codex 3차 adversarial-review 반영 — 기본값 변경이 조용한 의미 변경이 되지 않도록)."""
    kwargs.pop("position_method", None)
    return measure_pin(gray_full, position_method="auto", **kwargs)


def measure_pin_relative(gray_full: np.ndarray, **kwargs) -> dict:
    """`measure_pin(gray_full, position_method="pin_relative", ...)`의 명시적 별칭.
    이게 `measure_pin()`의 기본값과 같지만(E5 선정), 호출부가 "이 측정이 pin_relative를
    쓴다"는 걸 기본값에 기대지 않고 이름으로 분명히 하고 싶을 때 쓴다."""
    kwargs.pop("position_method", None)
    return measure_pin(gray_full, position_method="pin_relative", **kwargs)


def measure_pin_profile(gray_full: np.ndarray, margin_frac: float = 0.0, step: int = 2) -> dict:
    """핀 구간 전체(모따기·V블록 끝 그림자는 `coarse_section`의 높이 필터로 이미 제외됨)의
    열별 상·하단 엣지를 **cross50과 grad_peak 둘 다** 반환한다 (E5 전용).

    `measure_section`은 cross50 열별 값만 반환하고 grad_peak는 `d_peak_median`
    (전체 중앙값)으로만 요약하기 때문에, E5에서 cross50·grad_peak 기울기를 열별로
    각각 비교하려면 이 함수가 필요하다. `measure_section`과 똑같은 좌표·상수(coarse_section
    결과, 80px 창, step)를 쓰지만 기본 `margin_frac=0`(전체 구간)이라는 점만 다르다.
    로직이 `measure_section`과 겹치지만, E1에서 "수정하지 않는다"고 못 박은
    `measure_section`을 건드리지 않기 위해 별도 함수로 뒀다.
    """
    q = cv2.resize(gray_full, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    seg = coarse_section(q, **PIN_COARSE)
    if len(seg) < 6:
        raise ValueError(f"pin seg too short {len(seg)}")
    xa, xb = seg[0, 0] * 4, seg[-1, 0] * 4
    m = int((xb - xa) * margin_frac)
    xa, xb = xa + m, xb - m
    xs, top_cross, bot_cross, top_peak, bot_peak = [], [], [], [], []
    for X in range(xa, xb, step):
        si = np.argmin(np.abs(seg[:, 0] * 4 - X))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - 80, bc + 80
        prof = gray_full[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        t = edge_subpx(prof, tc - y0, 20, -1)
        b = edge_subpx(prof, bc - y0, 20, +1)
        xs.append(X)
        top_cross.append(t[1] + y0)
        bot_cross.append(b[1] + y0)
        top_peak.append(t[0] + y0)
        bot_peak.append(b[0] + y0)
    xs, top_cross, bot_cross, top_peak, bot_peak = map(np.array, (xs, top_cross, bot_cross, top_peak, bot_peak))
    return dict(
        x_from=int(xa), x_to=int(xb), n_cols=len(xs), x_per_col=xs,
        top_cross_per_col=top_cross, bot_cross_per_col=bot_cross,
        top_peak_per_col=top_peak, bot_peak_per_col=bot_peak,
        width_cross_per_col=bot_cross - top_cross, width_peak_per_col=bot_peak - top_peak,
    )


def extract_profiles(gray_full: np.ndarray, section: dict, half_window: int = 40, step: int = 2) -> np.ndarray:
    """상·하단 엣지 주변 밝기 프로파일을 열 단위로 뽑는다 (E7 데이터셋 빌더용).

    `section`은 `measure_bearing`/`measure_pin`이 반환한 dict(`top_y_per_col`,
    `bot_y_per_col`, `x_per_col` 포함)여야 한다. `step`은 `x_per_col`의 인덱스
    보폭이 아니라 **실제 이미지 x 픽셀 보폭**이다. `measure_section`이 이미
    `x_per_col`을 `step`(기본 2) px 간격으로 채워 놓으므로, 여기서도 같은
    간격(기본 2px)으로 x를 스캔하면서 `x_per_col`에서 가장 가까운 열을 찾아 쓴다.
    (수정 전 코드는 `x_per_col` 인덱스를 다시 `step`칸씩 건너뛰어 사실상 2×step
    px 간격이 되는 버그가 있었다 — E1 Codex 리뷰 반영, 재발 방지용 검증은
    `E1_module/reproduce.py`의 `check_extract_profiles_coverage`에 있다.)

    반환값은 `shape = (n_selected_cols, 2, 2*half_window+1)` 배열이다
    (axis 1: 0=상단, 1=하단). 길이가 짧은 영상 경계열은 제외한다.
    """
    xs = section["x_per_col"]
    tops = section["top_y_per_col"]
    bots = section["bot_y_per_col"]
    length = 2 * half_window + 1
    h = gray_full.shape[0]
    x_from, x_to = int(xs[0]), int(xs[-1])
    profiles = []
    for X in range(x_from, x_to + 1, step):
        k = int(np.argmin(np.abs(xs - X)))
        tc, bc = int(round(tops[k])), int(round(bots[k]))
        if tc - half_window < 0 or bc + half_window + 1 > h:
            continue
        col = gray_full[:, X - 1:X + 2].astype(np.float64).mean(1)
        top_prof = col[tc - half_window:tc + half_window + 1]
        bot_prof = col[bc - half_window:bc + half_window + 1]
        if len(top_prof) != length or len(bot_prof) != length:
            continue
        profiles.append(np.stack([top_prof, bot_prof]))
    if not profiles:
        return np.empty((0, 2, length), dtype=np.float64)
    return np.stack(profiles)
