"""PREREG_E13 2.3·3절: 창, 점 추출, 추세 제거, 지표. E11 detrend(robust_fit)를 import해 그대로 쓴다."""
import itertools
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from paths import EDGES, ROUGH, META, E11_CODE, SEED, MM_PER_PX

def _load_e11():
    """E11 코드도 paths.py를 쓰므로, 임시로 E13 paths를 치우고 E11 코드를 import한다(수정·복사 없음)."""
    mine = sys.modules.pop("paths")
    sys.path.insert(0, str(E11_CODE))
    try:
        import e11a_edge_texture as m
    finally:
        sys.path.remove(str(E11_CODE)); sys.modules["paths"] = mine
    return m


_e11 = _load_e11()
_e11_detrend = _e11.detrend

SPECS = [f"NO{k}" for k in range(2, 13)]
LEFT_PX, RIGHT_PX = 2 / MM_PER_PX, 3 / MM_PER_PX  # 76.65, 114.98
MIN_COLS = 30


def load():
    """본시험 110장 열별 엣지 + 이미지 표(시편, 파일, 시각순 번호, c)."""
    e = pd.read_csv(EDGES)
    e = e[(e["단계"] == "본시험(암막)") & e.specimen.isin(SPECS)]
    m = pd.read_csv(META, encoding="utf-8-sig")
    m = m[(m["단계"] == "본시험(암막)") & m.specimen.isin(SPECS)].sort_values(["specimen", "시각", "filename"])
    m["idx"] = m.groupby("specimen").cumcount()
    m["c"] = (m.bearing_x_from + m.bearing_x_to) / 2
    first_c = m[m.idx == 0].set_index("specimen")["c"]
    m["c1"] = m.specimen.map(first_c)
    return e, m.reset_index(drop=True), first_c


def window_H():
    return 8090.67, 8224.0


def window_C(c):
    return c - LEFT_PX, c + RIGHT_PX


def window_Crev(c):
    return c - RIGHT_PX, c + LEFT_PX


def cols(g, win, col="y_bot"):
    """창 안(경계 포함) 열: x, y."""
    x = g["x"].to_numpy(float); y = g[col].to_numpy(float)
    k = (x >= win[0]) & (x <= win[1])
    return x[k], y[k]


def pts(g, win, col="y_bot"):
    """x_k = 8008 + k*272/30 중 창 안의 점, np.interp로 y 보간."""
    x, y = g["x"].to_numpy(float), g[col].to_numpy(float)
    xk = 8008 + np.arange(31) * 272 / 30
    xk = xk[(xk >= win[0]) & (xk <= win[1])]
    return xk, np.interp(xk, x, y)


def trend(x, y, how):
    """none / poly1(최소제곱 직선) / poly2(E11 robust_fit). 잔차 반환."""
    if how == "none":
        return y
    if how == "poly1":
        u = (x - 8100.0) / 100.0
        return y - np.polyval(np.polyfit(u, y, 1), u)
    if how == "poly2":
        return _e11_detrend(x, y, 2)
    raise ValueError(how)


def metrics(d):
    med = np.median(d)
    return dict(MAD_med=float(np.median(np.abs(d - med))), MAD_mean=float(np.mean(np.abs(d - med))),
                STD=float(np.std(d, ddof=1)), Range=float(d.max() - d.min()))


def cv_diam(g, win):
    """창 안 열들의 median(y_bot - y_top). 시편별 10장 std/mean*100 은 호출측."""
    x = g["x"].to_numpy(float); k = (x >= win[0]) & (x <= win[1])
    return float(np.median((g["y_bot"] - g["y_top"]).to_numpy()[k]))


def cov(a, b):
    return float(np.cov(a, b, ddof=1)[0, 1])


def parity(g):
    return int(g["x"].min() % 2)


def decompose(res):
    """res: {spec: [(x, r) x10]} -> 시편별 V1, R, F, 쌍 수. 같은 절대 x, 홀짝 다른 쌍·공통 열 <30 쌍 제외."""
    def pair_cov(a, b):
        (xa, ra), (xb, rb) = a, b
        com = np.intersect1d(xa, xb)
        if len(com) < MIN_COLS:
            return None
        return cov(ra[np.isin(xa, com)], rb[np.isin(xb, com)])
    out = {}
    for p in res:
        V1 = float(np.mean([np.var(r, ddof=1) for _, r in res[p]]))
        w = [pair_cov(res[p][i], res[p][j]) for i, j in itertools.combinations(range(len(res[p])), 2)]
        w = [v for v in w if v is not None]
        fq = []
        for q in res:
            if q == p:
                continue
            c = [pair_cov(a, b) for a in res[p] for b in res[q]]
            c = [v for v in c if v is not None]
            if c:
                fq.append(np.mean(c))
        out[p] = dict(V1=V1, R=float(np.mean(w)) if w else np.nan, n_within_pairs=len(w),
                      F=float(np.median(fq)) if fq else np.nan, n_cross_q=len(fq))
    return out


def perm_spearman(x, y, n=10000, seed=SEED):
    """양측 순열: p = (|rho_perm| >= |rho_obs| 횟수 + 1)/(n + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("perm_spearman: 길이 불일치/3 미만/비유한 값")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")  # 상수 입력은 상관이 정의되지 않는다
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    cnt = sum(abs(spearmanr(x, rng.permutation(y))[0]) >= abs(rho) - 1e-12 for _ in range(n))
    return float(rho), (cnt + 1) / (n + 1)
