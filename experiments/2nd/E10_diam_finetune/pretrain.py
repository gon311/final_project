"""E10 1단계: fold별 사전학습(누수 방지).

`E7_edge1d/checkpoints/*.pt`는 정상 10시편 전체로 학습한 최종 모델이라 held-out
시편의 프로파일을 이미 봤다 — **시작점으로 쓰지 않는다**(지시서 E10 1단계). 시편
단위 LOSO 10 fold마다, 그리고 NG 평가용 "정상 10시편 전체" 모델 하나(`ALL`)까지
합쳐 11회, E7과 같은 설정(heatmap 헤드, 상단/하단 raw, 200 epoch, 시드 42, E7 허용
증강)으로 teacher(cross50) 복제 학습을 다시 한다. 하단 flipped 변형은 생략한다
(E7에서 heatmap raw와 차이가 0.005px, `E7_edge1d/README.md` 결과 요약 2 참고).

체크포인트는 `cache/E10_diam_finetune/pretrain_<시편>_{top,bot}.pt`(all-normal은
`<시편>="ALL"`)에 저장하고 커밋하지 않는다. **파일이 이미 있으면 건너뛴다**(지시서) —
재실행해도 끝낸 fold는 다시 학습하지 않는다.

같은 실행에서 fold별 **사전학습 OOF 예측**(해당 fold를 뺀 모델로 그 fold 자신을 예측)도
`cache/E10_diam_finetune/pretrain_oof.npz`에 모아 둔다 — E10 6단계(하단 단 이동 진단)가
"사전학습 예측"으로 쓰고, finetune.py의 teacher 유지 항(λ)도 이 체크포인트의 예측을
기준으로 쓴다.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import data as D  # 먼저 import: sys.path에 E7_edge1d/E6_calibration을 추가해 둔다
import augment as A  # noqa: E402  (E7, import만)
import model as M  # noqa: E402  (E7)
import train_eval as T  # noqa: E402  (E7, import만)

EDGES = ("top", "bottom")
HEAD = "heatmap"
EPOCHS = 200
SEED = 42


def ckpt_path(specimen: str, edge: str) -> str:
    tag = "top" if edge == "top" else "bot"
    return os.path.join(D.CACHE_DIR, f"pretrain_{specimen}_{tag}.pt")


def load_pretrained(specimen: str, edge: str) -> M.EdgeCNN1D:
    net = M.EdgeCNN1D(head=HEAD, half_window=A.HALF_WINDOW_PX).to(T.DEVICE)
    net.load_state_dict(torch.load(ckpt_path(specimen, edge), map_location=T.DEVICE))
    net.eval()
    return net


def _profiles_labels(cache, edge, mask):
    prof = cache["top_profiles"] if edge == "top" else cache["bot_profiles"]
    label = cache["top_label"] if edge == "top" else cache["bot_label"]
    return prof[mask], label[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    D.add_io_args(ap)  # --profiles/--meta/--pin-col/--label-source/--cache-dir/--results-dir
    args = ap.parse_args()

    cache, row_df = D.resolve_io_args(args)  # 디렉터리 전환 + 메타 CSV 기준 로드(Codex [high] 반영)
    role = row_df["role"].to_numpy()
    specimen = row_df["specimen"].to_numpy()
    normal_mask = role != "ng"
    normal_specimens = D.normal_specimens(row_df)  # 하드코딩 안 함(지시서 8번)

    sanity_rows = []
    oof = {}  # (specimen, edge) -> (row_idx_in_cache, pred)

    for edge in EDGES:
        t_edge0 = time.time()
        for held_out in normal_specimens + ["ALL"]:
            cp = ckpt_path(held_out, edge)
            train_mask = normal_mask if held_out == "ALL" else normal_mask & (specimen != held_out)

            if os.path.exists(cp):
                print(f"[skip, 이미 있음] {cp}")
            else:
                Xtr, ytr = _profiles_labels(cache, edge, train_mask)
                t0 = time.time()
                net = T.train_net(Xtr, ytr, edge=edge, head=HEAD, epochs=args.epochs, seed=SEED)
                torch.save(net.state_dict(), cp)
                print(f"[학습] {cp} (n={len(Xtr)}, {round(time.time()-t0,1)}s)", flush=True)

            if held_out != "ALL":
                # OOF: 이 fold를 뺀 모델로 held_out 시편 자신을 예측(teacher 복제 sanity check)
                held_mask = normal_mask & (specimen == held_out)
                Xte, yte = _profiles_labels(cache, edge, held_mask)
                net_eval = load_pretrained(held_out, edge)
                pred = T.predict(net_eval, A.center_crop(Xte))
                err = np.abs(pred - yte)
                sanity_rows.append(dict(edge=edge, specimen=held_out, n=len(Xte),
                                         median_abs_err_px=float(np.median(err)),
                                         max_abs_err_px=float(np.max(err))))
                oof[(held_out, edge)] = (np.flatnonzero(held_mask), pred)
        print(f"{edge}: {round(time.time()-t_edge0,1)}s", flush=True)

    # fold별 OOF 예측을 cache/에 모아 저장 (step_diag.py, finetune.py 재사용, 커밋 안 함)
    full_n = len(row_df)
    oof_top = np.full(full_n, np.nan, dtype=np.float32)
    oof_bot = np.full(full_n, np.nan, dtype=np.float32)
    for (sp, edge), (idx, pred) in oof.items():
        (oof_top if edge == "top" else oof_bot)[idx] = pred
    np.savez(os.path.join(D.CACHE_DIR, "pretrain_oof.npz"), top_pred=oof_top, bot_pred=oof_bot)

    sanity_df = pd.DataFrame(sanity_rows)
    os.makedirs(D.RESULTS_DIR, exist_ok=True)
    sanity_path = os.path.join(D.RESULTS_DIR, "pretrain_sanity.csv")
    sanity_df.to_csv(sanity_path, index=False, encoding="utf-8-sig")
    print()
    print("=== fold별 사전학습 OOF median|오차|(px) — E7 cv_metrics.csv 범위(top 0.008~0.035,")
    print("    bottom/raw 0.012~0.059)와 같은 범위인지 sanity check ===")
    print(sanity_df.groupby("edge")["median_abs_err_px"].agg(["mean", "min", "max"]).round(4))
    print(f"저장: {sanity_path}")


if __name__ == "__main__":
    main()
