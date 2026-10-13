"""하단(및 상단) 다단 엣지 진단용 저수준 함수 (E3 전용).

`measure_section`/`_measure_window`는 cross50 엣지 하나(전역 gradient 최댓값)만 다루므로
다단 분석에는 쓸 수 없다 — 이 모듈은 그 둘을 건드리지 않고 별도로 둔 것이다. 수식은
`edge_subpx`(E1, `measure.py`, 원본 그대로·수정 안 함)와 같은 방식(gradient 극댓값 포물선
보간, 50% 레벨 교차, 10~90% 엣지 폭)을 쓰되, 전역 최댓값 하나가 아니라 **극댓값 상위 n개**를
찾는다는 점만 다르다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _local_maxima_indices(arr: np.ndarray) -> list:
    """1차원 배열에서 이산 극댓값의 인덱스를 찾는다(양 끝 제외)."""
    idx = []
    for k in range(1, len(arr) - 1):
        if arr[k] >= arr[k - 1] and arr[k] >= arr[k + 1] and (arr[k] > arr[k - 1] or arr[k] > arr[k + 1]):
            idx.append(k)
    return idx


def find_peaks_subpixel(prof: np.ndarray, center: int, win: int, sign: int, n_peaks: int = 3,
                         min_sep: int = 3, cluster_radius: int = 22):
    """[center-win, center+win] 구간(지시서 0.4절대로 ≥40px)에서 `sign*gradient`의 극댓값
    상위 `n_peaks`개를 찾는다. 각 극댓값은 `edge_subpx`와 같은 포물선 보간으로 서브픽셀
    위치를 구한다. 반환: [(위치, 세기), ...] — **위치 오름차순**(물리적 순서, 1단/2단/3단 순).

    바깥 창은 넓게(≥40px, 배경/안쪽 밝기 샘플링에 필요) 두되, 실제 "몇 단인가" 판정은
    **가장 강한 극댓값에서 `cluster_radius`(기본 22px, 계단 전체 폭 20~25px + 여유) 안의
    극댓값만** 후보로 본다 — 그보다 멀리 있는 극댓값(베어링부 내부 반사띠 등 계단과
    무관한 밝기 변화)이 "3번째 단"으로 잘못 끼어드는 것을 막는다. `min_sep` 미만으로
    가까운 극댓값은 더 강한 쪽만 남긴다(같은 단의 흔들림 중복 방지).

    **`seg[k] > 0`인 극댓값만 후보로 삼는다(Codex 5차 review 반영).** `seg = g*sign`이
    음수(또는 0)인 지점의 "극댓값"은 이 엣지 방향(sign)과 반대로 기울어진 구간에서 그나마
    덜 음수인 점일 뿐, 진짜 단이 아니다 — 커밋된 데이터에서 3단으로 잡힌 15,376열 중
    1,884열(12.3%)이 이런 음수/0 세기 "단"을 포함하고 있었고, 이게 weakest/strongest
    비율(0단과 비교해 음수가 나오는 등)과 `n_peaks_detected`를 오염시켰다.
    """
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof)
    seg = g[a:b] * sign
    local_idx = [k for k in _local_maxima_indices(seg) if seg[k] > 0]
    if not local_idx:
        return []

    strongest = max(local_idx, key=lambda k: seg[k])
    candidates = [k for k in local_idx if abs(k - strongest) <= cluster_radius]
    candidates.sort(key=lambda k: -seg[k])  # 세기 내림차순

    selected = []
    for k in candidates:
        if all(abs(k - s) >= min_sep for s in selected):
            selected.append(k)
        if len(selected) >= n_peaks:
            break

    peaks = []
    for k in selected:
        ii = k + a
        g1, g2, g3 = g[ii - 1] * sign, g[ii] * sign, g[ii + 1] * sign
        den = g1 - 2 * g2 + g3
        off = 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (g1 - g3) / den, -1, 1))
        peaks.append((ii + off, float(seg[k])))
    peaks.sort(key=lambda p: p[0])
    return peaks


def edge_multistep_features(prof: np.ndarray, center: int, win: int, sign: int, n_peaks: int = 3,
                             ambiguous_dist_px: float = 2.0) -> dict:
    """한 열의 다단 엣지 특징을 모두 뽑는다.
      - peaks: [(위치, 세기), ...] 위치 오름차순, 최대 n_peaks개 (`find_peaks_subpixel`)
      - n_peaks_detected: 실제로 찾은 극댓값 개수. **주의(Codex 4차 adversarial-review 반영):**
        prominence/노이즈 바닥 기준이 없어 2~3이라는 숫자 자체가 "진짜 단 개수"를 보장하지
        않는다 — 핀처럼 절벽 하나인 구간도 거의 항상 3이 나온다(E3 README "방법론 메모").
        실제로 다단인지는 `peaks`의 세기(약한 쪽/강한 쪽 비율)로 따로 판단해야 한다.
      - cross50: 50% 레벨 교차 위치(`edge_subpx`와 동일 정의, **가장 강한** 극댓값 기준 bg/fg)
      - cross50_step: cross50이 `peaks` 중 어느 것에 가장 가까운지(1-based, 물리적 순서).
      - cross50_step_ambiguous: cross50이 가장 가까운 `peaks` 원소에서도 `ambiguous_dist_px`
        (기본 2px, 실제 데이터의 분포로 고름 — 대부분 <1px, 상위 1~3%만 이 문턱을 넘음)보다
        멀면 True. True인 열은 "어느 단이 cross50인가"를 신뢰하지 말아야 한다(Codex 4차
        adversarial-review 반영 — cross가 선택된 피크들 사이에 끼거나 진짜 전이 피크가
        상위 n_peaks에서 탈락한 경우를 걸러낸다).
      - bg_level, fg_level: 배경/안쪽 밝기(`edge_subpx`와 동일 정의)
      - edge_width_10_90, edge_width_90_to_50: 10~90% 폭, 90%→50%(비대칭) 폭
    """
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof)
    seg = g[a:b] * sign

    peaks = find_peaks_subpixel(prof, center, win, sign, n_peaks=n_peaks)
    if not peaks:
        return dict(peaks=[], n_peaks_detected=0, cross50=np.nan, cross50_step=np.nan,
                    cross50_step_ambiguous=True,
                    bg_level=np.nan, fg_level=np.nan, edge_width_10_90=np.nan, edge_width_90_to_50=np.nan)

    # 가장 강한 극댓값(전역 argmax, edge_subpx와 같은 기준)으로 bg/fg/cross50/엣지폭을 구한다.
    i = int(np.argmax(seg)) + a
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

    lo, hi = fg + 0.1 * (bg - fg), fg + 0.9 * (bg - fg)
    mid = fg + 0.5 * (bg - fg)
    r = prof[max(0, i - 30):i + 30]
    if sign < 0:
        r = r[::-1]
    try:
        k10 = int(np.flatnonzero(r >= lo)[0])
        k90 = int(np.flatnonzero(r >= hi)[0])
        k50 = int(np.flatnonzero(r >= mid)[0])
        ew_10_90 = float(k90 - k10)
        ew_90_50 = float(k90 - k50)
    except IndexError:
        ew_10_90 = ew_90_50 = np.nan

    nearest_idx = min(range(len(peaks)), key=lambda idx: abs(peaks[idx][0] - cross))
    nearest_dist = abs(peaks[nearest_idx][0] - cross)
    cross50_step = nearest_idx + 1
    cross50_step_ambiguous = bool(nearest_dist > ambiguous_dist_px)

    return dict(
        peaks=peaks, n_peaks_detected=len(peaks),
        cross50=float(cross), cross50_step=cross50_step,
        cross50_step_ambiguous=cross50_step_ambiguous,
        bg_level=float(bg), fg_level=float(fg),
        edge_width_10_90=ew_10_90, edge_width_90_to_50=ew_90_50,
    )


def scan_section_multistep(full: np.ndarray, seg: np.ndarray, xa: int, xb: int,
                            step: int = 2, half_window: int = 45, n_peaks: int = 3) -> list:
    """`coarse_section` 결과(`seg`)로 [xa, xb) 구간을 `step` 간격으로 훑어 각 열의
    상·하단 다단 엣지 특징을 반환한다(E3 전용). `_measure_window`와 열 추출 방식은
    같지만(같은 좌표계·블러), cross50 하나가 아니라 `edge_multistep_features`를 쓴다는
    점만 다르다 — `measure_section`/`_measure_window`는 그대로 둔다(E1 전제 유지).
    반환: [{x, top: {...}, bot: {...}}, ...]
    """
    rows = []
    for X in range(xa, xb, step):
        si = int(np.argmin(np.abs(seg[:, 0] * 4 - X)))
        tc, bc = int(seg[si, 2] * 4), int(seg[si, 3] * 4)
        y0, y1 = tc - half_window - 20, bc + half_window + 20
        prof = full[y0:y1, X - 1:X + 2].astype(np.float64).mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 1.2).ravel()
        top = edge_multistep_features(prof, tc - y0, half_window, -1, n_peaks)
        bot = edge_multistep_features(prof, bc - y0, half_window, +1, n_peaks)
        # 전역 좌표로 되돌린다(y0을 더함)
        for d in (top, bot):
            d["peaks"] = [(pos + y0, strength) for pos, strength in d["peaks"]]
            if not np.isnan(d["cross50"]):
                d["cross50"] += y0
        rows.append(dict(x=X, top=top, bot=bot))
    return rows
