"""E12 공용: 열별 엣지 추적(E11 e11a_endspike.py와 같은 산술), 사진 읽기(EXIF 회전 적용)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from paths import PHOTOS, RES1021

Image.MAX_IMAGE_PIXELS = None
LEVEL, W, TOL, CROP_DX = 105, 125, 3, 2408
meta = pd.read_csv(RES1021 / "meta_1021_all.csv", encoding="utf-8-sig")
main = meta[meta["단계"].str.startswith("본시험")]
POS = ["NO3", "NO5", "NO7", "NO9", "NO10"]
ROWS = {"zoom1": (2800, 4320), "NO1": (1700, 3300)}  # 윗엣지 첫 교차를 찾을 행 범위(원본 좌표, E6 COARSE_ROWS*4 / NO1 오프셋 반영)


def first_cross(col, lo=0):
    k = np.flatnonzero((col[lo:-1] >= LEVEL) & (col[lo + 1:] < LEVEL))
    if not len(k): return None
    k = k[0] + lo
    return k + (col[k] - LEVEL) / (col[k] - col[k + 1])


def window_edges(g, xc, rows):
    """xc±W 창의 열별 엣지 y. 창 중앙 3열 평균으로 대략 위치를 잡고 열마다 ±20px 안에서 교차를 찾는다."""
    r0, r1 = rows
    y0 = first_cross(g[r0:r1, xc - 1:xc + 2].mean(1))
    if y0 is None: return None
    y0 = int(r0 + y0)
    out = []
    for x in range(xc - W, xc + W):
        col = g[y0 - 20:y0 + 21, x]
        e = first_cross(col)
        out.append(np.nan if e is None else e + y0 - 20)
    e = np.array(out)
    return e - np.nanmedian(e)


def load(row):
    return np.asarray(ImageOps.exif_transpose(Image.open(PHOTOS / row.folder / row.filename)).convert("L"), float)
