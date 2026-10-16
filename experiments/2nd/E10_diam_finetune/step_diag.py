"""E10 6단계: 하단 단 이동 진단.

하단 프로파일마다 `shaftlab.multistep.find_peaks_subpixel`(E3과 같은 설정: n_peaks=3,
`win=40`, `sign=+1`)로 단 위치를 구한다. teacher(cross50 라벨), 사전학습 OOF 예측
(`pretrain.py`가 저장한 `pretrain_oof.npz`), 파인튜닝 OOF 예측(`finetune.py`가 저장한
`finetune_oof.npz`, 8변형)이 각각 — 중심(인덱스 44, 89px 프로파일 기준) 기준 오프셋 —
어느 단에 제일 가까운지 센다. 가장 가까운 단에서도 2px 넘게 떨어지면 `ambiguous`로
표시한다(E3 `edge_multistep_features`와 같은 문턱).

판단 기준(지시서): 실측 레이블(파인튜닝)이 하단 엣지를 **셋업과 관계없이 같은 단으로
일관되게** 옮기면 "외곽 후보에 대한 간접 증거", **셋업마다 다른 방향으로** 옮기면
"셋업 잡음 흡수"다 — `step_occupancy.csv`의 시편별 분포 흩어짐으로 이 판단을 내린다
(판단 자체는 README에 적는다).
"""
import argparse
import os

import numpy as np
import pandas as pd

import data as D
from shaftlab.multistep import find_peaks_subpixel

CENTER = 44  # 89px 저장 프로파일의 중심(half_window=40 + shift 버퍼 4)
WIN = 40
SIGN = 1  # 하단: dark->bright
N_PEAKS = 3
AMBIGUOUS_DIST_PX = 2.0


def nearest_step(value: float, peaks: list):
    if np.isnan(value) or not peaks:
        return np.nan, True
    dists = [abs(p[0] - CENTER - value) for p in peaks]
    k = int(np.argmin(dists))
    return k + 1, dists[k] > AMBIGUOUS_DIST_PX


def main():
    ap = argparse.ArgumentParser()
    D.add_io_args(ap)  # --profiles/--meta/--pin-col/--label-source/--cache-dir/--results-dir
    args = ap.parse_args()

    cache, row_df = D.resolve_io_args(args)
    normal_mask = (row_df["role"] != "ng").to_numpy()
    n = len(row_df)

    print("단 위치 계산 중(find_peaks_subpixel, 전체 행)...", flush=True)
    peaks_per_row = [find_peaks_subpixel(cache["bot_profiles"][i], CENTER, WIN, SIGN, n_peaks=N_PEAKS)
                      for i in range(n)]

    sources = {"teacher": cache["bot_label"].astype(np.float64)}

    pretrain_oof_path = os.path.join(D.CACHE_DIR, "pretrain_oof.npz")
    if os.path.exists(pretrain_oof_path):
        sources["pretrain"] = np.load(pretrain_oof_path)["bot_pred"].astype(np.float64)

    finetune_oof_path = os.path.join(D.CACHE_DIR, "finetune_oof.npz")
    if os.path.exists(finetune_oof_path):
        ft = np.load(finetune_oof_path)
        for key in ft.files:
            if key.endswith("__bot"):
                sources[key[: -len("__bot")]] = ft[key].astype(np.float64)

    rows = []
    specimen = row_df["specimen"].to_numpy()
    focus = cache["focus"] if "focus" in cache else np.array(["-"] * n)  # 다음 촬영 캐시엔 없을 수 있다
    for src_name, values in sources.items():
        for i in range(n):
            if not normal_mask[i] and src_name not in ("teacher",):
                continue  # OOF 소스는 NG 행에 값이 없다(LOSO로 안 뽑힘)
            step, ambiguous = nearest_step(values[i], peaks_per_row[i])
            rows.append(dict(source=src_name, specimen=specimen[i], focus=focus[i],
                              n_peaks_detected=len(peaks_per_row[i]),
                              step=("ambiguous" if ambiguous else step)))

    detail = pd.DataFrame(rows)
    occ = (detail.groupby(["source", "specimen", "focus", "step"]).size()
           .rename("n").reset_index())
    totals = occ.groupby(["source", "specimen", "focus"])["n"].transform("sum")
    occ["fraction"] = occ["n"] / totals

    os.makedirs(D.RESULTS_DIR, exist_ok=True)
    occ_path = os.path.join(D.RESULTS_DIR, "step_occupancy.csv")
    occ.to_csv(occ_path, index=False, encoding="utf-8-sig")
    print(f"저장: {occ_path} ({len(occ)}행)")

    pd.set_option("display.width", 220)
    print()
    print("=== source별 전체 step 점유율(시편·초점 무시, 참고) ===")
    overall = (detail.groupby(["source", "step"]).size().rename("n").reset_index())
    overall["fraction"] = overall["n"] / overall.groupby("source")["n"].transform("sum")
    print(overall.pivot(index="source", columns="step", values="fraction").round(3).to_string())


if __name__ == "__main__":
    main()
