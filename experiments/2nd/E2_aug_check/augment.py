"""E2 전용 증강 함수. 전부 원본 해상도(16320x7532) `uint8` 그레이스케일 이미지를 받아
같은 크기의 `float64` 배열을 돌려준다(`measure_bearing`이 내부에서 다시 캐스팅하므로
이대로 넘겨도 된다). `shaftlab`은 건드리지 않는다 — 증강은 이 실험 전용 전처리일 뿐이고
측정 함수(`coarse_section`/`edge_subpx`/`measure_section`)는 원본 그대로 쓴다.
"""
from __future__ import annotations

import cv2
import numpy as np


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img.astype(np.float64), (0, 0), sigma)


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img.astype(np.float64) + delta, 0, 255)


def contrast(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(128.0 + (img.astype(np.float64) - 128.0) * factor, 0, 255)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    x = img.astype(np.float64) / 255.0
    return np.clip(np.power(x, g) * 255.0, 0, 255)


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    """이미지 중심 기준 회전(원본 해상도, Lanczos 보간). "같은 위치 크롭"은 이 함수
    자체가 입력과 같은 크기(`(w, h)`)로 돌려주는 것으로 충족한다 — 어차피 호출부가
    원본과 같은 좌표(열 X, 중심 tc/bc)로 프로파일을 다시 뽑으므로 별도 크롭이 필요 없다."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img.astype(np.float64), m, (w, h), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REPLICATE)


def subpixel_shift_y(img: np.ndarray, dy: float) -> np.ndarray:
    """세로(상·하단 엣지와 수직인 방향) 서브픽셀 평행이동. 정답 이동량이 `dy`로 알려진
    대조군 — 상단·하단 둘 다 정확히 `dy`px만큼 움직여야 하고 지름(Δ외경)은 0이어야 한다."""
    m = np.float32([[1, 0, 0], [0, 1, dy]])
    h, w = img.shape
    return cv2.warpAffine(img.astype(np.float64), m, (w, h), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REPLICATE)


def pixel_average_downsample(img: np.ndarray, n: int, phase: tuple = (0, 0)) -> np.ndarray:
    """`n`x`n` 화소 평균 축소를 흉내 내되, **원본과 같은 크기로 다시 늘려(nearest 반복)
    돌려준다** — 그래야 좌표 상수(`BEARING_COARSE` 등)를 그대로 쓸 수 있다. 저해상도
    센서가 만드는 블러·앨리어싱만 재현하고, 화각·좌표계는 원본과 같게 유지하는 설계다.
    `phase`(px, py)는 축소 격자가 시작하는 화소 오프셋 — 팀원B가 본 "격자를 세로로
    1px만 옮겨도 오차가 크게 바뀐다"(0.2b)는 효과를 재현하려고 넣었다.
    """
    px, py = phase
    h, w = img.shape
    cropped = img[py:, px:].astype(np.float64)
    h2, w2 = (cropped.shape[0] // n) * n, (cropped.shape[1] // n) * n
    cropped = cropped[:h2, :w2]
    blocks = cropped.reshape(h2 // n, n, w2 // n, n).mean(axis=(1, 3))
    up = np.repeat(np.repeat(blocks, n, axis=0), n, axis=1)
    out = img.astype(np.float64).copy()
    out[py:py + h2, px:px + w2] = up
    return out
