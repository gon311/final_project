"""E11-A 사후 민감도 분석 (Codex 리뷰 지적 3건). 사전 등록 결과(E11A_*.csv)는 건드리지 않고 나란히 낸다.
실행: /usr/bin/python3 e11a_sensitivity.py
변형 축: matched(within·cross를 쌍별 교집합 열·같은 추세 제거 영역에서 계산) x trim(양 끝 열 제외) x NO8 처리 + 동등 필터."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from paths import PROFILES, RESULTS, SEED, UM_PER_PX
from e11a_edge_texture import (main, detrend, cov, spearman_perm, COMBOS, SPLITS_B, CROSS_IDX, MIN_CROSS_COLS,
                               SPECS, DESC)

rough = pd.read_csv(RESULTS / "roughness_clean.csv").set_index("specimen_1021")
RAW = {}
for s in SPECS:
    L = [(g["x"].to_numpy(), g["y_top"].to_numpy()) for _, g in main[main.specimen == s].groupby("filename", sort=True)]
    assert len(L) == 10
    RAW[s] = L


def prep(spec, trim, filt):
    out = []
    for x, y in RAW[spec]:
        if filt:  # 모든 이미지에 같은 2탭 평균([.5,.5], 1px 이동) — 동등 필터 변형
            assert (np.diff(x) == 2).all()
            x, y = x[:-1] + 1, (y[:-1] + y[1:]) / 2
        if trim:
            x, y = x[trim:len(x) - trim], y[trim:len(y) - trim]
        out.append((x, y))
    return out


def grid_of(imgs, phase):
    inter = sorted(set.intersection(*[set(x) for x, _ in imgs]))
    n_orig = int(np.median([len(x) for x, _ in imgs]))
    if len(inter) >= 0.8 * n_orig:
        return np.array(inter), False
    lo, hi = max(x.min() for x, _ in imgs), min(x.max() for x, _ in imgs)
    return np.arange(lo + (1 if phase == "B" else 0), hi + 1, 2), True


def on_cols(imgs, cols, interp):
    """원 y를 열 cols에 맞춰 (10, n) 행렬로, 거기서 다시 강건 2차 추세 제거(matched 모드)."""
    Y = np.stack([np.interp(cols, x, y) if interp else y[np.isin(x, cols)] for x, y in imgs])
    return np.stack([detrend(cols.astype(float), y, 2) for y in Y])


def within(R):
    return np.array([cov(R[list(a)].mean(0), R[list(b)].mean(0)) for a, b in zip(COMBOS, SPLITS_B)])


def cross(Rp, Rq):
    return float(np.median([cov(Rp[list(COMBOS[k])].mean(0), Rq[list(SPLITS_B[k])].mean(0)) for k in CROSS_IDX]))


def run(matched, trim, no8, filt):
    specs = [s for s in SPECS if not (no8 == "excl" and s == "NO8")]
    P = {s: prep(s, trim, filt) for s in specs}
    G = {s: grid_of(P[s], "B" if (s == "NO8" and no8 == "B") else "A") for s in specs}
    if not matched:  # 사전 등록 추정량: 시편 자기 격자에서 within, 쌍별 교집합에서 cross, 추세 제거는 이미지 원래 열 전체
        R = {}
        for s in specs:
            grid, interp = G[s]
            R[s] = np.stack([np.interp(grid, x, detrend(x.astype(float), y, 2)) if interp
                             else detrend(x.astype(float), y, 2)[np.isin(x, grid)] for x, y in P[s]])
    rows = []
    for p in specs:
        Sw_all, Sc_pairs, dW = None, [], []
        if not matched:
            Sw = within(R[p])
        for q in specs:
            if q == p: continue
            C = np.intersect1d(G[p][0], G[q][0])
            if len(C) < MIN_CROSS_COLS: continue
            if matched:
                Rp, Rq = on_cols(P[p], C, G[p][1]), on_cols(P[q], C, G[q][1])
                Swq = within(Rp); Sc = cross(Rp, Rq)
                Sw_all = Swq if Sw_all is None else np.vstack([Sw_all, Swq])
                dW.append(Swq - Sc); Sc_pairs.append(Sc)
            else:
                Sc_pairs.append(cross(R[p][:, np.isin(G[p][0], C)], R[q][:, np.isin(G[q][0], C)]))
        Sc_med = float(np.median(Sc_pairs))
        if matched:
            S_spec = float(np.median([np.median(d) for d in dW])); p05 = float(np.percentile(np.median(np.stack(dW), 0), 5))
            Sw_med = float(np.median(Sw_all))
        else:
            S_spec = float(np.median(Sw)) - Sc_med; p05 = float(np.percentile(Sw - Sc_med, 5)); Sw_med = float(np.median(Sw))
        a = np.sqrt(max(S_spec, 0))
        rows.append(dict(specimen=p, n_grid_cols=len(G[p][0]), interpolated=G[p][1], n_cross_pairs=len(Sc_pairs),
                         S_within_med=Sw_med, S_cross_med=Sc_med, S_spec=S_spec, a_spec_um=a * UM_PER_PX, p05=p05,
                         has_specific=p05 > 0, Rz=rough.at[p, "Rz"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    variants = [(m, t, n, False) for m in (False, True) for t in (0, 3, 6) for n in ("A", "B", "excl")]
    variants += [(m, t, "excl", True) for m in (False, True) for t in (0, 3, 6)]
    base = None; by, summ = [], []
    for m, t, n, f in variants:
        d = run(m, t, n, f)
        name = f"{'matched' if m else 'prereg'}|trim{t}|NO8_{n}|{'filt2' if f else 'nofilt'}"
        d.insert(0, "variant", name); by.append(d)
        if (m, t, n, f) == (False, 0, "A", False):
            base = d.set_index("specimen")
            ref = pd.read_csv(RESULTS / "E11A_by_specimen.csv"); ref = ref[(ref.edge == "top") & (ref.detrend == "poly2")].set_index("specimen")
            assert np.allclose(base.S_spec, ref.S_spec.loc[base.index], atol=1e-9), "사전 등록 결과 재현 실패"
        rho, p = spearman_perm(d.Rz.to_numpy(), d.a_spec_um.to_numpy())
        n_has = int(d.has_specific.sum()); need = 6 if len(d) == 11 else None
        stage1 = n_has >= 6  # PREREG 기준 그대로(시편 10개일 때도 6 유지)
        case = 3 if not stage1 else (1 if (rho > 0 and p < 0.05) else 2)
        base_a = base.a_spec_um.reindex(d.specimen).to_numpy(); ok = ~np.isnan(base_a)
        summ.append(dict(variant=name, matched=m, trim=t, NO8=n, equal_filter=f, n_spec=len(d), n_has_specific=n_has,
                         no_specific="/".join(d.specimen[~d.has_specific]), rho_Rz=rho, p_one_sided=p, conclusion_case=case,
                         rho_a_vs_prereg=float(np.corrcoef(rankdata(d.a_spec_um[ok]), rankdata(base_a[ok]))[0, 1])))
    pd.concat(by).to_csv(RESULTS / "E11A_sens_by_specimen.csv", index=False)
    sm = pd.DataFrame(summ); sm.to_csv(RESULTS / "E11A_sens_summary.csv", index=False)

    # 끝열 진단: 캐시 프로파일(윗엣지)로 끝열의 대비·엣지 오프셋을 내부 열과 비교
    z = np.load(PROFILES, allow_pickle=True)
    fn, sp, x = z["filename"].astype(str), z["specimen"].astype(str), z["x"]
    prof, lab = z["top_profiles"].astype(float), z["top_label"]
    isMain = np.isin(fn, main.filename.unique())
    er = []
    for s in SPECS:
        m_ = isMain & (sp == s); xs = np.unique(x[m_])
        def stat(sel):
            return (np.median(np.median(prof[sel][:, :20], 1) - np.median(prof[sel][:, -20:], 1)), np.median(lab[sel]))
        c0, l0 = stat(m_ & (x >= xs[10]) & (x <= xs[-11]))
        for k in range(5):
            c, l = stat(m_ & (x == xs[-1 - k]))
            er.append(dict(specimen=s, from_right_end=k, x=int(xs[-1 - k]), contrast_gray=c, contrast_interior=c0,
                           label_offset_px=l, label_interior_px=l0))
    pd.DataFrame(er).to_csv(RESULTS / "E11A_endcols.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(sm.round(3).to_string(index=False))
    e = pd.DataFrame(er); print(e[e.from_right_end <= 1].round(1).to_string(index=False))
