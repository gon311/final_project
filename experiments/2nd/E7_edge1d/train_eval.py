"""E7 학습·평가: 상단/하단 각각 1D CNN(회귀/heatmap 헤드)을 GroupKFold(LOSO, 셋업=시편
단위, 규칙 3)로 학습·평가한다.

- **teacher = cross50**(E4 1순위 후보, `E4_edge_benchmark/README.md`). `build_cache.py`가
  저장한 `top_label`/`bot_label`(프로파일 중심 기준 서브픽셀 오프셋)이 라벨이다.
  **중심은 teacher가 아니라 `coarse_section`의 1/4 해상도 coarse 추정이다**(Codex
  adversarial-review [high] 반영) — 처음 버전은 중심을 `round(teacher)`로 잡아 라벨이
  반올림 오차(±0.5px)뿐인 순환 과제였다. 지금 라벨은 상단 -5~0px, 하단 +5~+15px 범위로
  훨씬 넓고, teacher와 무관한 중심에서 실제 엣지를 "찾는" 과제가 됐다(`build_cache.py`
  docstring 참고). **profiles.npz의 `top_profiles`/`bot_profiles`는 89px(모델 입력
  81px + shift 보간용 버퍼 4px 양쪽)로 저장돼 있다** — 이 파일의 모든 함수가 예측 직전에
  `A.center_crop`(또는 `A.augment_batch`/`A.apply_eval_augment`, 둘 다 내부에서 자동으로
  자른다)으로 81px로 줄인 뒤에만 네트워크에 넣는다.
- **분할은 시편 단위**다. `profiles.npz`의 `specimen` 필드가 이미 NO1(묶음1+13 포함)을
  하나로 묶어 주므로(0.3절 분류표 자체가 그렇게 돼 있음) 추가 병합 로직이 필요 없다.
  정상 10개 시편(NO1~NO10)에 대해 `LeaveOneGroupOut` -> 정확히 10-fold LOSO다.
- **NG1·NG2는 학습·검증 fold에 전혀 넣지 않는다(규칙 4)** — "all-normal"(10개 시편 전체로
  학습한 최종 체크포인트)에만 넣어 평가용으로 한 번 돌린다(`fold="ng_eval_only"`로 표시).
- **베이스라인(규칙 10 취지를 이 작업에 맞게 번역):** 이 실험엔 "실측 평균 출력" 같은
  mm 베이스라인이 없다(라벨이 px 서브픽셀 오프셋이라 mm 실측과 직접 대응하지 않는다).
  대신 "오프셋을 아예 안 보정한다"(항상 0 예측, 즉 `edge_subpx`의 정수 중심 그대로)를
  null 모델로 같이 낸다 — CNN이 이 null보다 못하면 "이 창에서 학습이 정보를 못 담는다"로
  본다.
- **하단은 좌우 반전(방향을 상단과 맞춤) 버전도 비교한다**(지시서 3번) — `--variant flipped`.
- **증강은 `augment.py`(E2 허용 목록)로 학습 배치마다 무작위로 적용**한다. 평가는 깨끗한
  라벨(clean)과 `augment.EVAL_INTENSITIES`의 고정 강도들로 둘 다 낸다(강건성 표).
- **외경 재계산 σ**: 같은 행(이미지의 같은 열)에 상단·하단 라벨이 짝지어 저장돼 있으므로
  `diam = (bot_center+bot_label) - (top_center+top_label)`로 계산한다. 상단 모델만 있을
  때는 하단을 teacher 그대로 두고 상단만 예측값으로 바꿔 짐(그 반대도 마찬가지) — "상단만
  바꿨을 때 셋업 내 분산이 teacher 대비 어떻게 변하는가"를 보는 설계다.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import augment as A  # noqa: E402
import model as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(HERE, "..", "cache", "E7_edge1d", "profiles.npz")
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS_DIR = os.path.join(HERE, "results")

SEED = 42
PHYSICAL_SCALE_MM_PER_PX = 0.0367  # 0.2절, 참고 환산용
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


EXPECTED_PROFILE_WIDTH = A.MODEL_WIDTH + 2 * A.SHIFT_BUFFER_PX  # 81 + 2*4 = 89


def load_cache():
    """Codex adversarial-review [P2, 2026-10-16] 반영: 89px 포맷 이전의 81px
    `profiles.npz`를 그대로 두고 재실행하면, `A.crop_center`가 `[4:85]`로 잘라
    77px짜리를 조용히 모델에 넣어 학습·평가가 틀어진 채로 성공해버린다(에러 없음).
    캐시 폭을 여기서 검증해 그 경우 바로 실패시킨다."""
    d = np.load(CACHE_NPZ, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    for key in ("top_profiles", "bot_profiles"):
        width = out[key].shape[-1]
        if width != EXPECTED_PROFILE_WIDTH:
            raise RuntimeError(
                f"{CACHE_NPZ}의 '{key}' 폭이 {width}px이다 (기대값 {EXPECTED_PROFILE_WIDTH}px "
                f"= 모델 입력 {A.MODEL_WIDTH}px + shift 버퍼 {A.SHIFT_BUFFER_PX}px 양쪽). "
                "89px 포맷 이전에 만든 캐시로 보인다 - 캐시를 지우고 "
                "`python3 build_cache.py`로 다시 만든 뒤 재실행하라."
            )
    return out


def train_net(profiles, labels, edge, head, epochs=None, seed=SEED):
    """`profiles`는 `build_cache.py`가 저장한 넓은(89px) 배열이다 — 모델이 실제로 보는
    폭(81px = `A.HALF_WINDOW_PX`)은 `A.augment_batch`가 잘라서 돌려준다(shift 외
    증강은 버퍼를 버리고 중앙만 쓴다)."""
    if epochs is None:
        epochs = EPOCHS
    set_seed(seed)
    net = M.EdgeCNN1D(head=head, half_window=A.HALF_WINDOW_PX).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    n = len(profiles)
    idx_all = np.arange(n)
    net.train()
    for _ in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, n, BATCH_SIZE):
            batch_idx = idx_all[s:s + BATCH_SIZE]
            bp_aug, bl_aug = A.augment_batch(profiles[batch_idx], labels[batch_idx], edge, rng)
            xb = torch.tensor(bp_aug, dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(bl_aug, dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, profiles, batch_size=1024):
    outs = []
    for s in range(0, len(profiles), batch_size):
        xb = torch.tensor(profiles[s:s + batch_size], dtype=torch.float32, device=DEVICE)
        outs.append(net(xb).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty((0,), dtype=np.float32)


def maybe_flip(profiles, labels, variant):
    if variant == "flipped":
        return profiles[:, ::-1].copy(), -labels
    return profiles, labels


def unflip_pred(pred, variant):
    return -pred if variant == "flipped" else pred


def diam_from(top_center, top_val, bot_center, bot_val):
    return (bot_center + bot_val) - (top_center + top_val)


def setup_sigma_table(df, value_col):
    """셋업(시편)별 이미지 대표값(열 중앙값) -> 셋업 내 10장 사이 표준편차."""
    per_image = df.groupby(["specimen", "filename"])[value_col].median().reset_index()
    per_setup_sigma = per_image.groupby("specimen")[value_col].std(ddof=0)
    return per_setup_sigma


def run_edge(edge: str, head: str, variant: str, cache: dict, rows_df: pd.DataFrame):
    """edge in {top,bottom}, head in {reg,heatmap}, variant in {raw,flipped}(bottom만 의미 있음).
    LOSO 10-fold로 학습·평가하고 (oof_pred, metrics_rows, final_net)을 반환한다."""
    label_col = "top_label" if edge == "top" else "bot_label"
    profiles_all = cache["top_profiles"] if edge == "top" else cache["bot_profiles"]
    labels_all = cache[label_col]
    specimen_all = cache["specimen"]
    role_all = cache["role"]

    normal = role_all != "ng"
    groups = specimen_all[normal]
    X = profiles_all[normal]
    y = labels_all[normal]
    idx_normal = np.flatnonzero(normal)

    logo = LeaveOneGroupOut()
    oof_pred = np.full(len(X), np.nan, dtype=np.float32)
    metrics_rows = []

    for fold_i, (tr, te) in enumerate(logo.split(X, y, groups=groups)):
        held_out_specimen = groups[te][0]
        Xtr, ytr = maybe_flip(X[tr], y[tr], variant)
        net = train_net(Xtr, ytr, edge=edge, head=head, seed=SEED)

        # clean 평가(held-out) — X[te]는 넓은(89) 배열이라 예측 전에 중앙 81px로 자른다
        Xte_v, yte_v = maybe_flip(X[te], y[te], variant)
        pred_te = unflip_pred(predict(net, A.center_crop(Xte_v)), variant)
        oof_pred[te] = pred_te
        err = np.abs(pred_te - y[te])
        metrics_rows.append(dict(
            edge=edge, head=head, variant=variant, fold=held_out_specimen, aug="clean", intensity=np.nan,
            n=len(te), mae_px=float(np.median(err)), max_abs_err_px=float(np.max(err)),
        ))

        # baseline(오프셋 항상 0)
        err0 = np.abs(0.0 - y[te])
        metrics_rows.append(dict(
            edge=edge, head="baseline_zero", variant=variant, fold=held_out_specimen, aug="clean", intensity=np.nan,
            n=len(te), mae_px=float(np.median(err0)), max_abs_err_px=float(np.max(err0)),
        ))

        # 증강 강도별 강건성(held-out에만, 정답은 라벨칙에 따라 shift만 이동)
        for kind, inten in A.EVAL_INTENSITIES[edge]:
            if kind == "clean":
                continue
            p_aug, l_aug = A.apply_eval_augment(X[te], y[te], kind, inten)
            p_aug_v, l_aug_v = maybe_flip(p_aug, l_aug, variant)
            pred_aug = unflip_pred(predict(net, p_aug_v), variant)
            err_aug = np.abs(pred_aug - l_aug)
            metrics_rows.append(dict(
                edge=edge, head=head, variant=variant, fold=held_out_specimen, aug=kind, intensity=inten,
                n=len(te), mae_px=float(np.median(err_aug)), max_abs_err_px=float(np.max(err_aug)),
            ))

    # 전체(10개 시편)로 학습한 최종(deliverable) 체크포인트
    Xall, yall = maybe_flip(X, y, variant)
    final_net = train_net(Xall, yall, edge=edge, head=head, seed=SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_name = f"{edge}_{head}_{variant}.pt"
    torch.save(final_net.state_dict(), os.path.join(CKPT_DIR, ckpt_name))

    # NG1·NG2 평가(학습에는 전혀 안 씀, 규칙 4)
    ng_mask = role_all == "ng"
    if ng_mask.any():
        Xng, yng = maybe_flip(profiles_all[ng_mask], labels_all[ng_mask], variant)
        pred_ng = unflip_pred(predict(final_net, A.center_crop(Xng)), variant)
        err_ng = np.abs(pred_ng - labels_all[ng_mask])
        metrics_rows.append(dict(
            edge=edge, head=head, variant=variant, fold="ng_eval_only(all-normal 모델)", aug="clean", intensity=np.nan,
            n=int(ng_mask.sum()), mae_px=float(np.median(err_ng)), max_abs_err_px=float(np.max(err_ng)),
        ))

    oof_full = np.full(len(profiles_all), np.nan, dtype=np.float32)
    oof_full[idx_normal] = oof_pred
    return oof_full, metrics_rows, final_net


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("device:", DEVICE, flush=True)
    cache = load_cache()
    n_rows = len(cache["filename"])
    rows_df = pd.DataFrame(dict(
        filename=cache["filename"], specimen=cache["specimen"], bundle=cache["bundle"],
        role=cache["role"], x=cache["x"],
        top_center=cache["top_center"], bot_center=cache["bot_center"],
        top_label=cache["top_label"], bot_label=cache["bot_label"],
    ))

    all_metrics = []
    oof = {}

    combos = [
        ("top", "reg", "raw"), ("top", "heatmap", "raw"),
        ("bottom", "reg", "raw"), ("bottom", "reg", "flipped"),
        ("bottom", "heatmap", "raw"), ("bottom", "heatmap", "flipped"),
    ]
    for edge, head, variant in combos:
        t0 = time.time()
        oof_pred, metrics_rows, _ = run_edge(edge, head, variant, cache, rows_df)
        oof[(edge, head, variant)] = oof_pred
        all_metrics += metrics_rows
        print(f"{edge}/{head}/{variant}: {round(time.time()-t0,1)}s", flush=True)

    # oof 예측을 따로 저장(cache/, 커밋 안 함) — sigma 재조립처럼 재학습 없이 후처리만
    # 바꾸고 싶을 때 재사용한다.
    oof_cache_path = os.path.join(HERE, "..", "cache", "E7_edge1d", "oof_predictions.npz")
    np.savez(oof_cache_path, **{f"{e}__{h}__{v}": arr for (e, h, v), arr in oof.items()})

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "cv_metrics.csv"), index=False)

    # 셋업 내 sigma 비교(지름 재계산) — Stage A: 상단만 예측값, 하단은 teacher
    sigma_rows = []
    teacher_diam = diam_from(rows_df["top_center"].values, rows_df["top_label"].values,
                              rows_df["bot_center"].values, rows_df["bot_label"].values)
    rows_df["diam_teacher"] = teacher_diam
    normal = (rows_df["role"] != "ng").values

    def add_sigma(label, diam_values):
        tmp = rows_df.copy()
        tmp["diam_tmp"] = diam_values
        sig = setup_sigma_table(tmp[normal], "diam_tmp")
        for specimen, s in sig.items():
            sigma_rows.append(dict(model=label, specimen=specimen, sigma_px=float(s)))

    add_sigma("teacher(cross50 전체)", teacher_diam)

    for head in ("reg", "heatmap"):
        pred_top = oof[("top", head, "raw")]
        diam_predtop = diam_from(rows_df["top_center"].values, pred_top,
                                  rows_df["bot_center"].values, rows_df["bot_label"].values)
        add_sigma(f"stageA_top_{head}(하단=teacher)", diam_predtop)

        for variant in ("raw", "flipped"):
            pred_bot = oof[("bottom", head, variant)]
            diam_predbot = diam_from(rows_df["top_center"].values, rows_df["top_label"].values,
                                      rows_df["bot_center"].values, pred_bot)
            add_sigma(f"stageA_bottom_{head}_{variant}(상단=teacher)", diam_predbot)

    # Stage B(탐색): 상단·하단 둘 다 모델 예측 — 측정 정확도로 해석하지 않음.
    # heatmap·reg 둘 다 낸다 — clean MAE에서 heatmap이 reg를 압도적으로 이겨서
    # (아래 "결과 요약"), 헤드 선택이 지름 재계산 σ에도 그대로 영향을 주는지 보려는
    # 의도다.
    for head in ("heatmap", "reg"):
        pred_top_h = oof[("top", head, "raw")]
        pred_bot_h = oof[("bottom", head, "raw")]
        diam_both = diam_from(rows_df["top_center"].values, pred_top_h,
                               rows_df["bot_center"].values, pred_bot_h)
        add_sigma(f"stageB_both_{head}_raw(탐색, 정확도 아님)", diam_both)

    sigma_df = pd.DataFrame(sigma_rows)
    sigma_df.to_csv(os.path.join(RESULTS_DIR, "setup_sigma.csv"), index=False)

    print()
    print("=== 셋업 내 sigma(px) 요약 (낮을수록 좋음, teacher와 비교) ===")
    print(sigma_df.groupby("model")["sigma_px"].agg(["mean", "max"]).round(4))

    print()
    print("=== clean mae(px) 요약 (median, fold 평균) ===")
    clean = metrics_df[(metrics_df["aug"] == "clean") & (metrics_df["fold"] != "ng_eval_only(all-normal 모델)")]
    print(clean.groupby(["edge", "head", "variant"])["mae_px"].mean().round(4))

    print()
    print(f"저장: {os.path.join(RESULTS_DIR, 'cv_metrics.csv')}")
    print(f"저장: {os.path.join(RESULTS_DIR, 'setup_sigma.csv')}")


if __name__ == "__main__":
    main()
