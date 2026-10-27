"""E11-0 재현 확인 (a)~(d) -> results/E11_0_checks.csv. 거칠기와 이미지 지표는 비교하지 않는다."""
import numpy as np
import pandas as pd
from paths import PROFILES, RES1021, RESULTS, CACHE

rows = []
def add(check, item, value, criterion="", ok=""):
    rows.append(dict(check=check, item=item, value=value, criterion=criterion, ok=ok))

z = np.load(PROFILES, allow_pickle=True)
edges = pd.read_csv(CACHE / "edges_long.csv.gz")
vals = pd.read_csv(RES1021 / "values_by_image_1021.csv", encoding="utf-8-sig").set_index("filename")

# ---- (a) 이미지별 1차 강건적합 중심값 vs bearing_top_y / bearing_bot_y
def robust_fit(x, y):  # measure.py _measure_window 안의 robust_fit과 동일
    p = np.polyfit(x, y, 1)
    r = y - np.polyval(p, x)
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    k = np.abs(r) < 4 * 1.4826 * mad
    return np.polyfit(x[k], y[k], 1)

dt, db, dn = [], [], []
for fn, g in edges.groupby("filename", sort=False):
    x = g["x"].to_numpy(float); xc = x.mean()
    ft = np.polyval(robust_fit(x, g["y_top"].to_numpy()), xc)
    fb = np.polyval(robust_fit(x, g["y_bot"].to_numpy()), xc)
    dt.append(ft - vals.at[fn, "bearing_top_y"]); db.append(fb - vals.at[fn, "bearing_bot_y"])
    dn.append(len(g) - vals.at[fn, "bearing_n_cols"])
dt, db, dn = map(np.abs, (np.array(dt), np.array(db), np.array(dn)))
n_img = len(dt)
add("a", "이미지 수", n_img, "140", n_img == 140)
add("a", "|top 차이| 최대 px", dt.max(), "< 0.01", dt.max() < 0.01)
add("a", "|top 차이| 중앙값 px", np.median(dt))
add("a", "|bot 차이| 최대 px", db.max(), "< 0.01", db.max() < 0.01)
add("a", "|bot 차이| 중앙값 px", np.median(db))
add("a", "열 수 차이(캐시-bearing_n_cols) 최대", dn.max(), "0", dn.max() == 0)

# ---- (b) 프로파일 인덱스 i <-> y = center - 44 + i : cross50 재계산 vs label
def gauss7(p, sigma=1.2):  # cv2.GaussianBlur((1,7), 1.2)와 같은 커널, 경계는 BORDER_REFLECT_101
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(p, 3, mode="reflect"), k, mode="valid")

def cross50(prof, center, win, sign):  # measure.py edge_subpx의 cross 부분(원본 복사, numpy만 사용)
    a, b = max(2, center - win), min(len(prof) - 2, center + win)
    g = np.gradient(prof)
    i = int(np.argmax(g[a:b] * sign)) + a
    if sign < 0:
        bg = np.median(prof[max(0, i - 45):max(1, i - 25)]); fg = np.median(prof[i + 25:i + 45])
    else:
        fg = np.median(prof[max(0, i - 45):max(1, i - 25)]); bg = np.median(prof[i + 25:i + 45])
    lvl = 0.5 * (bg + fg)
    j = i
    if sign < 0:
        while j > a and prof[j] < lvl: j -= 1
        while j < b and prof[j + 1] >= lvl: j += 1
    else:
        while j > a and prof[j] > lvl: j -= 1
        while j < b and prof[j + 1] <= lvl: j += 1
    p0, p1 = prof[j], prof[j + 1]
    return j + (0 if p1 == p0 else (lvl - p0) / (p1 - p0)), i

rng = np.random.default_rng(42)
for edge, sign in (("top", -1), ("bot", +1)):
    P = z[f"{edge}_profiles"].astype(np.float64); L = z[f"{edge}_label"]
    idx = rng.choice(len(P), 2000, replace=False)  # 전수는 불필요, 시드 42 무작위 2000열
    for name, sm in (("평활없음", False), ("Gaussian(1,7)σ1.2", True)):
        d, intact = [], []
        for n in idx:
            p = gauss7(P[n]) if sm else P[n]
            c, i = cross50(p, 44, 20, sign)
            d.append(c - 44 - L[n])
            # 원본 cross50은 [i-45:i-25]와 [i+25:i+45]를 모두 쓴다. 양쪽 구간(평활이면 커널 경계 3px 포함)이 캐시 안에 있어야 '잘림 없음'
            m = 3 if sm else 0
            intact.append(i - 45 >= m and i + 45 <= len(p) - m)
        d, intact = np.abs(np.array(d)), np.array(intact)
        add("b", f"{edge} {name} |cross50 재계산-label| 중앙값 px", np.median(d))
        add("b", f"{edge} {name} 95퍼센타일 px", np.percentile(d, 95))
        add("b", f"{edge} {name} 양쪽 샘플 구간이 온전한 열 수(2000열 중)", int(intact.sum()),
            "0이면 89px 캐시로는 원본 완전 재현 검증 불가")
        if intact.any():
            add("b", f"{edge} {name} 온전한 열 |차이| 중앙값 px", np.median(d[intact]))
        add("b", f"{edge} {name} |차이|<0.01px 비율", (d < 0.01).mean())

# ---- (c) E6 재현
e6 = pd.read_csv(RES1021 / "E6" / "calibration_by_specimen_1021.csv", encoding="utf-8-sig")
e6 = e6[(e6.정의 == "cross50") & (e6.사진선택 == "all") & (e6.분할 == "A_시편LOSO")]
e6 = e6[e6.unit.isin([f"NO{k}" for k in range(2, 11)])]
for mode, ref in (("0_평균찍기", 6.64), ("3_물리스케일고정(V1)", 8.48), ("4_핀비율정규화선형", 3.99)):
    s = e6[e6.방식 == mode]
    m = s.abs_err_um.mean()
    add("c", f"{mode} 정상 {len(s)}시편 MAE µm", m, f"{ref}(±0.005)", len(s) == 9 and abs(m - ref) < 0.005)

# ---- (d) 본시험 시편별 이미지 수와 x 교집합
main = edges[edges["단계"] == "본시험(암막)"]
for sp, g in main.groupby("specimen", sort=False):
    sets = [set(h["x"]) for _, h in g.groupby("filename")]
    inter = set.intersection(*sets); per = [len(s) for s in sets]
    add("d", f"{sp} 이미지 수 / 교집합 열 / 이미지당 열 최소~최대", f"{len(sets)} / {len(inter)} / {min(per)}~{max(per)}",
        "10장", len(sets) == 10)

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "E11_0_checks.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 80)
print(out.to_string(index=False))
