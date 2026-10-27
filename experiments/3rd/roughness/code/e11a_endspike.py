"""E11-A 끝열 원영상 확인: x≈8265 부근 엣지 급락이 시편 표면인지 사진(측정) 쪽 고정 패턴인지 본다.
원본 사진을 EXIF 회전까지 적용해 열고(OpenCV와 같은 방향) 열마다(1px) 엣지 위치를 직접 추적한다.
시편마다 사진 N_IMG장의 중앙값 프로파일에서, 주변(±6px) 중앙값 대비 가장 큰 편차 열을 찾는다.
shortcut: 시편당 앞 5장만 쓴다(장당 ~1초), 더 늘리려면 N_IMG만 바꾼다."""
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from paths import PROFILES, RES1021, RESULTS, UM_PER_PX

Image.MAX_IMAGE_PIXELS = None
PHOTOS = "/Volumes/Lexar/Final/_private/20261021_촬영데이터/"  # 저장소 밖, 읽기 전용
XS = np.arange(8240, 8300)
N_IMG, HALF, LEVEL = 5, 44, 105  # LEVEL: 밝음(~200)과 어두움(~10)의 중간 밝기


def edge_y(col, edge):
    """한 열의 엣지 y(서브픽셀). 윗엣지는 밝음->어두움, 아랫엣지는 어두움->밝음 교차."""
    hit = (col[:-1] >= LEVEL) & (col[1:] < LEVEL) if edge == "top" else (col[:-1] < LEVEL) & (col[1:] >= LEVEL)
    k = np.flatnonzero(hit)
    if not len(k): return np.nan
    k = k[0]
    return k + (col[k] - LEVEL) / (col[k] - col[k + 1])


z = np.load(PROFILES, allow_pickle=True)
fn, sp = z["filename"].astype(str), z["specimen"]
meta = pd.read_csv(RES1021 / "meta_1021_all.csv", encoding="utf-8-sig").set_index("filename")
rows = []
for s in sorted(set(sp), key=lambda t: int(t[2:])):
    P = {"top": [], "bot": []}
    for f in sorted(set(fn[sp == s]))[:N_IMG]:
        i = np.flatnonzero(fn == f)[0]
        g = np.asarray(ImageOps.exif_transpose(Image.open(PHOTOS + meta.loc[f, "folder"] + "/" + f)).convert("L"), float)
        for edge in P:
            c = int(z[f"{edge}_center"][i])
            e = np.array([edge_y(g[c - HALF:c + HALF + 1, x], edge) for x in XS])
            P[edge].append(e - np.nanmedian(e))
    for edge, D in P.items():
        d = np.nanmedian(D, 0)
        dev = d - pd.Series(d).rolling(13, center=True, min_periods=5).median().values
        j = np.nanargmax(np.abs(dev))
        rows.append(dict(specimen=s, edge=edge, n_img=len(D), x_peak=int(XS[j]), dev_px=round(dev[j], 2),
                         dev_um=round(dev[j] * UM_PER_PX), at_8262_8270=bool(8262 <= XS[j] <= 8270)))
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "E11A_endspike.csv", index=False)
print(out.to_string(index=False))
