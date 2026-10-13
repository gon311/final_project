"""E4 전용 엣지 정의 3·4·5 — `edge_subpx`(cross50/grad_peak, E1·원본 그대로·수정 안 함)에
없는 세 가지만 추가한다. `measure_section`/`_measure_window`/`edge_subpx`는 건드리지 않고
같은 좌표계(열 중심 프로파일, `win`은 중심에서 양쪽으로 몇 px 볼지)만 공유한다.

  - `level_crossing`: `edge_subpx`의 50% 교차를 임의 레벨로 일반화한 것(하단 문턱 sweep용).
    bg/fg 추정(그래디언트 최댓값에서 25~45px 떨어진 구간의 중앙값)은 `edge_subpx`와 동일하다.
  - `moment_centroid`: 그래디언트 최댓값 주변 `radius`px의 1차 모멘트(무게중심) 기반 서브픽셀
    위치. 포물선 보간(`edge_subpx`)과 달리 좌우 비대칭 그래디언트 모양에 덜 민감하다.
  - `esf_fit`: 열 프로파일에 n단(1~3) 로지스틱(ESF) 모델을 최소제곱(scipy)으로 맞춘다.
    팀원B M5("계단 모양 맞춤")와 같은 계열. 초기값은 `shaftlab.multistep.find_peaks_subpixel`
    (E3, 이미 seg[k]>0 필터로 검증됨)이 찾은 극댓값 위치를 쓴다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .multistep import find_peaks_subpixel


def _grad_peak_index(prof: np.ndarray, center: int, win: int, sign: int) -> tuple[int, np.ndarray]:
    """`edge_subpx`와 동일한 방식(전역 최강 gradient)으로 엣지 중심 인덱스를 찾는다."""
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof)
    seg = g[a:b] * sign
    i = int(np.argmax(seg)) + a
    return i, g


def _bg_fg(prof: np.ndarray, i: int, sign: int) -> tuple[float, float]:
    """`edge_subpx`와 동일한 bg/fg 샘플링(엣지에서 25~45px 떨어진 구간 중앙값)."""
    if sign < 0:
        bg = np.median(prof[max(0, i - 45):max(1, i - 25)])
        fg = np.median(prof[i + 25:i + 45])
    else:
        fg = np.median(prof[max(0, i - 45):max(1, i - 25)])
        bg = np.median(prof[i + 25:i + 45])
    return float(bg), float(fg)


def level_crossing(prof: np.ndarray, center: int, win: int, sign: int, level: float = 0.5) -> float:
    """`edge_subpx`의 50% 교차를 임의 `level`(0~1, fg 쪽 0 ~ bg 쪽 1)로 일반화한다.
    `level=0.5`이면 `edge_subpx`의 cross50과 **값이 같다**(같은 bg/fg·같은 탐색 로직).
    E4 "하단 문턱 sweep"에서 상단은 0.5로 고정하고 하단만 0.1~0.9로 바꿔 부른다.
    """
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    i, _ = _grad_peak_index(prof, center, win, sign)
    bg, fg = _bg_fg(prof, i, sign)
    lvl = fg + level * (bg - fg)
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
    return float(j + (0 if p1 == p0 else (lvl - p0) / (p1 - p0)))


def moment_centroid(prof: np.ndarray, center: int, win: int, sign: int, radius: int = 15) -> float:
    """그래디언트 최댓값(`edge_subpx`와 같은 기준) 주변 ±`radius`px의 1차 모멘트(무게중심).
    포물선 보간 3점(`edge_subpx`)과 달리 비대칭 그래디언트 모양 전체를 반영한다.
    음수 가중치(반대 방향 기울기)는 0으로 자른다(E3 Codex 5차 review의 seg[k]>0 교훈과 같은
    이유 — 반대 방향 기울기는 이 엣지의 위치 정보가 아니다).
    """
    i, g = _grad_peak_index(prof, center, win, sign)
    lo, hi = max(1, i - radius), min(len(prof) - 1, i + radius)
    idx = np.arange(lo, hi + 1)
    w = np.clip(g[lo:hi + 1] * sign, 0, None)
    if w.sum() <= 0:
        return float(i)
    return float((idx * w).sum() / w.sum())


MIN_STEP_SEP = 1.5  # px. 적합된 n개 단 중 이보다 가까운 쌍이 있으면 "사실상 중복"으로 보고 실패 처리.


def _logistic_steps(x: np.ndarray, bg: float, fg: float, sign: int, centers: np.ndarray, sigma: float) -> np.ndarray:
    """n개 단을 가진 ESF 모델. 각 단이 전체 fg->bg 변화량을 동일하게 나눠 갖고, 같은 폭(sigma)을
    공유한다(팀원B M5처럼 "계단 모양"이되, 단마다 다른 폭을 맞출 자유도는 n=3·열당 데이터 양에
    비해 과하다고 보고 공유 폭으로 단순화했다 — 한계에 기록).
    """
    n = len(centers)
    amp = (bg - fg) / n
    out = np.full_like(x, fg, dtype=np.float64)
    for c in centers:
        step = 1.0 / (1.0 + np.exp(-sign * (x - c) / max(sigma, 1e-6)))
        out = out + amp * step
    return out


def esf_fit(prof: np.ndarray, center: int, win: int, sign: int, n_steps: int = 1) -> dict:
    """[center-win, center+win] 구간에 n단 로지스틱(ESF) 모델을 최소제곱으로 맞춘다.
    bg/fg는 `edge_subpx`와 같은 방식으로 고정하고(자유도를 줄여 n=3에서도 안정적인 적합이
    되게 함), 초기 단 위치는 `find_peaks_subpixel`(E3)이 찾은 극댓값으로 준다 — 진짜 단이
    있으면 그 근처에서 시작해 수렴이 더 안정적이다.

    반환: dict(centers=[...] n_steps개, 오름차순, sigma, success, rmse, min_gap). 실패(초기
    **방향성** 극댓값이 n_steps개 미만이거나 least_squares가 수렴 실패, 또는 **적합된 단
    n개 중 둘 이상이 서로 `MIN_STEP_SEP`px 안으로 붙어 사실상 중복된 경우**)하면
    `success=False`이고 `centers`는 `grad_peak` 위치로 n_steps번 채운 더미값이다(호출부가
    NaN 전파로 집계를 망치지 않게). **단 붕괴(collapse) 검사를 추가한 이유(Codex
    adversarial-review 반영):** `least_squares`의 `success`(수렴 종료 여부)만 보면 실패율이
    0%로 나왔는데도 — 즉 scipy는 "정상 종료"라고 보고하는데도 — 실제로는 n=3인데 진짜 단이
    1개뿐인 열에서 3개 중 2개가 거의 같은 위치(서로 <0.01px 차이)로 겹쳐버리고 나머지 1개가
    창 끝 쪽으로 멀리 밀려나는 "성공적으로 수렴한 퇴화해"가 흔했다(코드로 확인, 예: 한
    이미지에서 center 간격이 0.0003px인 열이 다수). 이게 esf2/esf3_outer의 큰 잔차·체계차의
    실제 원인이었다 — `success=False`를 그대로 뒀다면 이 문제를 놓쳤을 것이다.
    **`n_steps=1`일 때는 극댓값 부족을 scipy 호출 전에 바로 걸러낸다(Codex `/codex:review`
    반영):** 단 붕괴 검사는 center가 2개 이상이어야 뜻이 있다(`min_gap`을 비교할 쌍이
    있어야 함) — `n_steps=1`이면 `zip(centers, centers[1:])`가 항상 비어서 `min_gap=inf`가
    되고 단 붕괴 검사가 통과만 시킨다. 그런데 `find_peaks_subpixel`이 진짜 방향성 극댓값을
    **0개**밖에 못 찾은 열(그래디언트가 전부 반대 방향인 경우 등)도 이전 코드는
    `grad_peak` 위치로 초기값을 때워 `least_squares`를 그대로 돌렸고, scipy는 이런 가짜
    초기값에도 거의 항상 "수렴했다"고 보고해 `esf1_ok=True`로 잘못 기록됐다 — `esf1`의
    "실패율 0%"라는 수치 자체가 이 구멍 때문에 과소평가됐을 수 있다는 지적이었다. 이제
    `len(peaks) < n_steps`면 `least_squares`를 아예 돌리지 않고 즉시 실패 처리한다.
    """
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    i, _ = _grad_peak_index(prof, center, win, sign)
    bg, fg = _bg_fg(prof, i, sign)
    x = np.arange(a, b, dtype=np.float64)
    y = prof[a:b].astype(np.float64)

    peaks = find_peaks_subpixel(prof, center, win, sign, n_peaks=n_steps)
    if len(peaks) < n_steps:
        # 방향성 극댓값 자체가 모자라면(0개 포함) 가짜 초기값으로 fitting을 시도하지 않고
        # 바로 실패 처리한다 — scipy는 가짜 초기값에도 "수렴했다"고 보고하는 경우가 흔해서,
        # fitting을 돌리면 success가 거짓으로 True가 될 수 있다(Codex `/codex:review` 반영).
        c0 = sorted([p[0] for p in peaks] + [float(i)] * (n_steps - len(peaks)))
        return dict(centers=c0, sigma=float("nan"), success=False, rmse=float("nan"), min_gap=float("nan"))
    c0 = sorted(p[0] for p in peaks[:n_steps])
    p0 = np.array(c0 + [3.0])  # 마지막 파라미터 = sigma 초기값

    def resid(p):
        centers, sigma = p[:-1], p[-1]
        return _logistic_steps(x, bg, fg, sign, centers, sigma) - y

    lo = np.array([a] * n_steps + [0.3])
    hi = np.array([b] * n_steps + [15.0])
    try:
        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200)
        centers = sorted(float(c) for c in res.x[:-1])
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        min_gap = float(min((b2 - a2 for a2, b2 in zip(centers, centers[1:])), default=float("inf")))
        ok = bool(res.success) and min_gap >= MIN_STEP_SEP
        return dict(centers=centers, sigma=float(res.x[-1]), success=ok, rmse=rmse, min_gap=min_gap)
    except Exception:
        return dict(centers=sorted(c0), sigma=float("nan"), success=False, rmse=float("nan"), min_gap=float("nan"))
