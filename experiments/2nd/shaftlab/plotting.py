"""matplotlib 그림에 한글 라벨을 쓰기 위한 공용 헬퍼 (E5, Codex 2차 adversarial-review의
추가 요청: 한글 라벨이 □로 깨지는 문제 해결).

`setup_korean_font()`를 그림을 그리기 전에 호출한다. macOS 기준으로 AppleGothic ->
Apple SD Gothic Neo -> NanumGothic -> Malgun Gothic 순으로, 실제 `matplotlib`에 설치된
폰트(`font_manager.fontManager.ttflist`) 중에서만 고른다. 하나도 없으면 경고만 남기고
`rcParams`를 건드리지 않는다 — 그 경우 호출한 스크립트가 영어 라벨로 대체해야 한다.
"""
import warnings

import matplotlib
from matplotlib import font_manager

# 우선순위: macOS 기본(AppleGothic) -> macOS 대체(Apple SD Gothic Neo) -> 다른 OS에 흔한 한글 폰트
KOREAN_FONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]

_cached_font = None  # None=아직 안 찾음, ""=찾아봤는데 없음, 그 외=찾은 폰트 이름
_cached_available_names = None


def _available_font_names():
    global _cached_available_names
    if _cached_available_names is None:
        _cached_available_names = {f.name for f in font_manager.fontManager.ttflist}
    return _cached_available_names


def setup_korean_font() -> str | None:
    """한글 폰트를 찾아 `plt.rcParams`에 설정한다. 찾은 폰트 이름(설정 완료)을 반환하고,
    하나도 없으면 `None`을 반환하며 경고만 남긴다(rcParams는 바꾸지 않음 — 호출부가
    영어 라벨로 대체해야 한다는 신호)."""
    global _cached_font
    if _cached_font is not None:
        if _cached_font:
            matplotlib.rcParams["font.family"] = _cached_font
            matplotlib.rcParams["axes.unicode_minus"] = False
        return _cached_font or None

    available = _available_font_names()
    for name in KOREAN_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            _cached_font = name
            return name

    _cached_font = ""
    warnings.warn(
        "한글 폰트를 찾지 못했다(시도: " + ", ".join(KOREAN_FONT_CANDIDATES) + "). "
        "그림 라벨은 영어로 대체해야 한다(setup_korean_font()가 None을 반환함).",
        stacklevel=2,
    )
    return None
