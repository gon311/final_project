"""E10 2~5단계: 지름 파인튜닝(8변형) + 평가(중첩 LOSO).

**변형(지시서 4번):** V{1,2}(환산) x {H, H+B}(학습범위) x λ{0,1}(teacher 유지) = 8개.
- V1 물리 스케일: `pred_mm = 0.0367 * D + c`.
- V2 핀 비율: `pred_mm = 10 * D / pin_px + c`, 묶음1(P1)은 학습에서 뺀다(E6와 같은 처리).
- H: heatmap 헤드(1x1 conv)만 학습, 백본은 고정. H+B: 마지막 conv 블록(`conv4`/`bn4`)과
  헤드를 학습. BatchNorm은 두 경우 모두 `eval()`로 고정해 러닝 통계가 흔들리지 않게 한다
  (파인튜닝 대상은 가중치뿐 — 지시서가 명시하지 않은 설계 선택, README에 기록).
- `c`는 학습 스칼라(nn.Parameter), 초기값은 학습 fold의 teacher D로 구한 최소제곱 값.
- λ=1이면 `λ * (mean|top_pred-top_pretrain_pred| + mean|bot_pred-bot_pretrain_pred|)`(px
  단위)를 Huber(µm 단위) 손실에 그대로 더한다 — 단위를 맞추지 않고 지시서 식을 그대로
  따른다(px 스케일이 µm 손실보다 훨씬 작아 λ=1의 영향이 작을 수 있다는 점을 README에
  적는다).
- 배치: "이미지 단위로 묶는다"(지시서)를 fold의 모든 학습 열을 **한 번에** 넣는 full-batch
  GD로 구현한다(fold당 학습 이미지 ~90~100장, 미니배치로 나눌 필요가 없을 만큼 작다) —
  이미지별 절사평균(`data.trimmed_mean_rows`)이 손실 계산 단계에서 "이미지 단위로 묶는다"
  를 구현한다.

**중첩 LOSO(지시서 5번, 핵심):** fold k(held-out 시편 k)마다
  1. 파인튜닝·c는 **학습 9시편으로만** 맞춘다(held-out은 전혀 안 보임).
  2. "모델 직접 예측" 경로: 파인튜닝된 변환식(V1/V2 + c)으로 held-out 이미지의 pred_mm을
     직접 계산한다.
  3. "모델 px + 기존 캘리브레이션" 경로: 같은 파인튜닝 모델로 **학습 9시편의 D_i를 다시
     계산**해 E6 `calibration.py`(fit_const/fit_linear/fit_fixed_scale, import만)와 핀비율
     선형(E6 run_calibration.py 방식 재구현, 묶음1 제외)을 **fold 안에서** 맞추고 held-out
     을 예측한다 — out-of-fold px를 한꺼번에 모아 쓰지 않는다(지시서 경고).
- NG1·NG2는 "정상 10시편 전체로 파인튜닝한 모델"(held_out="ALL")로 평가만 한다.
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import data as D  # 먼저 import: sys.path에 E7_edge1d/E6_calibration을 추가해 둔다
import augment as A  # noqa: E402  (E7, import만)
import calibration as C  # noqa: E402  (E6, import만)
import train_eval as T  # noqa: E402  (E7, import만)
import pretrain as P

FT_EPOCHS = 50
FT_LR = 1e-4
SEED = 42
HUBER_DELTA_UM = 10.0  # 공차 반폭(10um, 0.3절)과 같은 크기로 선택 — README에 근거 기록
TRIM_FRAC = 0.1

V_SCALES = ("V1", "V2")
V_SCOPES = ("H", "HB")
LAMBDAS = (0.0, 1.0)
VARIANTS = list(itertools.product(V_SCALES, V_SCOPES, LAMBDAS))  # 8개

NG_COLUMNS = ("variant", "specimen", "eval_path", "method", "mm_true", "pred_mm", "err_um", "판정", "방향")


def variant_tag(v_scale, v_scope, lam) -> str:
    return f"{v_scale}_{v_scope}_l{int(lam)}"


def trainable_filter(scope: str):
    if scope == "H":
        return lambda name: name.startswith("head.")
    if scope == "HB":
        return lambda name: (name.startswith("head.") or name.startswith("backbone.conv4")
                              or name.startswith("backbone.bn4"))
    raise ValueError(scope)


def clone_trainable(held_out: str, edge: str, scope: str):
    """사전학습 체크포인트를 복제해 `scope`에 맞는 파라미터만 `requires_grad=True`로 둔다.
    BatchNorm 러닝 통계가 흔들리지 않게 `eval()`로 고정한다(위 모듈 docstring)."""
    net = P.load_pretrained(held_out, edge)
    filt = trainable_filter(scope)
    for name, p in net.named_parameters():
        p.requires_grad = filt(name)
    net.eval()
    return net


def convert_mm(D_img, pin_img, v_scale: str, c):
    if v_scale == "V1":
        return D.PHYSICAL_SCALE_MM_PER_PX * D_img + c
    if v_scale == "V2":
        return 10.0 * D_img / pin_img + c
    raise ValueError(v_scale)


def init_c(D_img_np: np.ndarray, mm_np: np.ndarray, pin_np: np.ndarray, v_scale: str) -> float:
    """c 초기값: 학습 fold의 teacher D로 구한 최소제곱(절편) 값 — `fit_fixed_scale`과
    같은 형태(평균 잔차)로 구한다(지시서: "D_i^teacher로 구한 최소제곱 값")."""
    if v_scale == "V1":
        pred0 = D.PHYSICAL_SCALE_MM_PER_PX * D_img_np
    else:
        pred0 = 10.0 * D_img_np / pin_np
    return float(np.mean(mm_np - pred0))


def run_fold(held_out: str, v_scale: str, v_scope: str, lam: float, cache: dict, row_df: pd.DataFrame,
             epochs: int = FT_EPOCHS, seed: int = SEED, verbose: bool = False,
             apply_1008_quirks: bool = True, include_pin_methods: bool = True):
    """한 (fold, 변형) 조합을 파인튜닝하고 평가 행들을 돌려준다.
    반환: dict(direct_row, calib_rows(list), edge_shift(dict), oof(dict top/bot idx,pred))
    held_out == "ALL"이면 평가 없이 파인튜닝만 하고 finetuned (top_net,bot_net,c)를 그대로
    돌려준다(NG 평가용).

    `apply_1008_quirks`(Codex adversarial-review [P2] 반영, 2026-10-16 2차): 묶음1(P1)
    V2 학습 제외와 핀비율선형(방식4)의 NO1 학습 제외는 둘 다 10-08 데이터(핀 재배치
    P1→P2) 특유의 규칙이다 - 기본값 True는 label_source="micrometer"(10-08)일 때만이고,
    main()이 label_source="gauge"면 False를 넘긴다. 다음 촬영에서 우연히 묶음 ID가
    1이거나 시편이 "NO1"이어도 더 이상 이유 없이 제외되지 않는다.
    `include_pin_methods`: pin_px가 메타에 없거나 일부 비어 있으면 main()이 False를
    넘긴다 - V2 환산과 방식4(핀비율선형)는 전부 pin_px로 나누므로, 그러지 않으면
    NaN이 조용히 결과에 섞여 들어간다(Codex [P2] 지적)."""
    torch.manual_seed(seed)
    role = row_df["role"].to_numpy()
    specimen = row_df["specimen"].to_numpy()
    bundle = row_df["bundle"].to_numpy()
    normal_mask = role != "ng"

    train_mask = normal_mask.copy() if held_out == "ALL" else normal_mask & (specimen != held_out)
    if v_scale == "V2" and apply_1008_quirks:
        train_mask = train_mask & (bundle != 1)  # 묶음1(P1) 학습 제외(10-08 전용, 지시서)

    idx_lists, img_df = D.image_table(row_df, train_mask)
    Xtop = A.center_crop(cache["top_profiles"][train_mask])
    Xbot = A.center_crop(cache["bot_profiles"][train_mask])
    ytop = cache["top_label"][train_mask]
    ybot = cache["bot_label"][train_mask]
    top_center = row_df.loc[train_mask, "top_center"].to_numpy(dtype=np.float32)
    bot_center = row_df.loc[train_mask, "bot_center"].to_numpy(dtype=np.float32)

    dev = T.DEVICE
    Xtop_t = torch.tensor(Xtop, dtype=torch.float32, device=dev)
    Xbot_t = torch.tensor(Xbot, dtype=torch.float32, device=dev)
    top_center_t = torch.tensor(top_center, device=dev)
    bot_center_t = torch.tensor(bot_center, device=dev)
    mm_img = img_df["실측mm"].to_numpy(dtype=np.float32)
    pin_img = img_df["pin_px"].to_numpy(dtype=np.float32)
    mm_img_t = torch.tensor(mm_img, device=dev)
    pin_img_t = torch.tensor(pin_img, device=dev)

    net_top = clone_trainable(held_out, "top", v_scope)
    net_bot = clone_trainable(held_out, "bottom", v_scope)

    with torch.no_grad():
        pretrain_top0 = net_top(Xtop_t).detach().clone()
        pretrain_bot0 = net_bot(Xbot_t).detach().clone()
        d_row_teacher = (bot_center_t + torch.tensor(ybot, device=dev)) - \
                        (top_center_t + torch.tensor(ytop, device=dev))
        D_img_teacher0 = D.trimmed_mean_rows(d_row_teacher, idx_lists, TRIM_FRAC).cpu().numpy()
    c0 = init_c(D_img_teacher0, mm_img, pin_img, v_scale)
    c = nn.Parameter(torch.tensor(c0, dtype=torch.float32, device=dev))

    trainable_params = [p for p in net_top.parameters() if p.requires_grad] + \
                        [p for p in net_bot.parameters() if p.requires_grad] + [c]
    opt = torch.optim.Adam(trainable_params, lr=FT_LR)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA_UM)

    for ep in range(epochs):
        opt.zero_grad()
        top_pred = net_top(Xtop_t)
        bot_pred = net_bot(Xbot_t)
        d_row = (bot_center_t + bot_pred) - (top_center_t + top_pred)
        D_img = D.trimmed_mean_rows(d_row, idx_lists, TRIM_FRAC)
        pred_mm = convert_mm(D_img, pin_img_t, v_scale, c)
        diam_loss = loss_fn(pred_mm * 1000.0, mm_img_t * 1000.0)
        loss = diam_loss
        if lam > 0:
            retention = (top_pred - pretrain_top0).abs().mean() + (bot_pred - pretrain_bot0).abs().mean()
            loss = loss + lam * retention
        loss.backward()
        opt.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep{ep}: diam_loss={diam_loss.item():.2f}um loss={loss.item():.2f} c={c.item():.4f}")

    net_top.eval()
    net_bot.eval()
    with torch.no_grad():
        top_pred_f = net_top(Xtop_t)
        bot_pred_f = net_bot(Xbot_t)
        d_row_f = (bot_center_t + bot_pred_f) - (top_center_t + top_pred_f)
        D_img_train = D.trimmed_mean_rows(d_row_f, idx_lists, TRIM_FRAC).cpu().numpy()

    out = dict(net_top=net_top, net_bot=net_bot, c=float(c.item()),
               img_df=img_df, D_img_train=D_img_train,
               top_pred_train=top_pred_f.cpu().numpy(), bot_pred_train=bot_pred_f.cpu().numpy(),
               train_mask=train_mask)

    if held_out == "ALL":
        return out

    # --- held-out 평가 (train_mask에서 전혀 보지 않은 시편의 전체 이미지, 변형 제외필터 미적용) ---
    held_mask = normal_mask & (specimen == held_out)
    idx_lists_h, img_df_h = D.image_table(row_df, held_mask)
    Xtop_h = A.center_crop(cache["top_profiles"][held_mask])
    Xbot_h = A.center_crop(cache["bot_profiles"][held_mask])
    top_center_h = torch.tensor(row_df.loc[held_mask, "top_center"].to_numpy(dtype=np.float32), device=dev)
    bot_center_h = torch.tensor(row_df.loc[held_mask, "bot_center"].to_numpy(dtype=np.float32), device=dev)
    Xtop_h_t = torch.tensor(Xtop_h, dtype=torch.float32, device=dev)
    Xbot_h_t = torch.tensor(Xbot_h, dtype=torch.float32, device=dev)
    with torch.no_grad():
        top_pred_h = net_top(Xtop_h_t)
        bot_pred_h = net_bot(Xbot_h_t)
        d_row_h = (bot_center_h + bot_pred_h) - (top_center_h + top_pred_h)
        D_img_h = D.trimmed_mean_rows(d_row_h, idx_lists_h, TRIM_FRAC).cpu().numpy()
    pin_img_h = img_df_h["pin_px"].to_numpy(dtype=np.float32)
    direct_pred_mm_h = convert_mm(D_img_h, pin_img_h, v_scale, float(c.item()))
    mm_true = float(img_df_h["실측mm"].iloc[0])
    direct_pred_mm_specimen = float(np.median(direct_pred_mm_h))
    err_um_direct = (direct_pred_mm_specimen - mm_true) * 1000.0

    # --- 중첩 캘리브레이션: 학습 9시편 D_i -> 시편표 -> E6 네 방식 ---
    calib_rows = []
    held_specimen_px = float(np.median(D_img_h))  # 물리/비율 스케일 전 px(지름)
    setup_px = D.specimen_table_from_images(img_df, D_img_train)  # 실측mm, px(지름)
    for method_name, fitter in (("0_평균찍기", C.fit_const), ("2_전체선형회귀", C.fit_linear),
                                 ("3_물리스케일고정", C.fit_fixed_scale)):
        predict_fn, params = fitter(setup_px)
        pred_mm = predict_fn(held_specimen_px)
        calib_rows.append(dict(method=method_name, pred_mm=pred_mm,
                                err_um=(pred_mm - mm_true) * 1000.0, slope=params.get("slope"),
                                intercept=params.get("intercept")))
    # 방식4: 핀비율 선형, NO1 학습 제외(10-08 전용, E6와 같은 처리) — pin_px가 없으면
    # (include_pin_methods=False) 통째로 건너뛴다(Codex [P2] 반영, NaN 전파 방지).
    if include_pin_methods:
        ratio_train = img_df.copy()
        ratio_train["ratio"] = D_img_train / img_df["pin_px"].to_numpy(dtype=np.float32)
        exclude = {"NO1"} if apply_1008_quirks else frozenset()
        setup_ratio = D.specimen_table_from_images(ratio_train, ratio_train["ratio"].to_numpy(),
                                                    exclude_specimens=exclude)
        predict_fn4, params4 = C.fit_linear(setup_ratio)
        held_ratio = float(np.median(D_img_h / pin_img_h))
        pred_mm4 = predict_fn4(held_ratio)
        calib_rows.append(dict(method="4_핀비율정규화선형", pred_mm=pred_mm4,
                                err_um=(pred_mm4 - mm_true) * 1000.0, slope=params4.get("slope"),
                                intercept=params4.get("intercept")))

    # --- 열별 엣지 이동량(teacher 대비, held-out 열 기준) ---
    ytop_h = cache["top_label"][held_mask]
    ybot_h = cache["bot_label"][held_mask]
    top_shift = np.abs(top_pred_h.cpu().numpy() - ytop_h)
    bot_shift = np.abs(bot_pred_h.cpu().numpy() - ybot_h)

    direct_row = dict(held_out=held_out, mm_true=mm_true, pred_mm=direct_pred_mm_specimen,
                       err_um=err_um_direct, abs_err_um=abs(err_um_direct))

    oof = dict(held_mask=held_mask, top_pred=top_pred_h.cpu().numpy(), bot_pred=bot_pred_h.cpu().numpy())
    edge_shift = dict(top_median=float(np.median(top_shift)), top_p95=float(np.percentile(top_shift, 95)),
                       bot_median=float(np.median(bot_shift)), bot_p95=float(np.percentile(bot_shift, 95)))

    return dict(direct_row=direct_row, calib_rows=calib_rows, edge_shift=edge_shift, oof=oof)


def judge_direction(mm: float):
    return C.judge(mm), C.direction(mm)


def ng_eval_rows_for(tag: str, v_scale: str, res_all: dict, row_df: pd.DataFrame, cache: dict,
                      apply_1008_quirks: bool = True, include_pin_methods: bool = True) -> list:
    """NG1·NG2 평가(정상 10시편 전체로 학습/파인튜닝한 `res_all`로 평가만, 규칙 4).
    `model_direct`(변환식+c 직접)과 `model_px+calib`(E6 네 방식, 학습 10시편으로 피팅)
    둘 다 돌린다. 변형 루프와 teacher 베이스라인 루프가 이 함수를 공유한다.
    `apply_1008_quirks`/`include_pin_methods`는 `run_fold`와 같은 뜻이다(Codex
    adversarial-review [P2] 반영, 2026-10-16 2차)."""
    rows = []
    setup_px_all = D.specimen_table_from_images(res_all["img_df"], res_all["D_img_train"])
    predict_fn4 = None
    if include_pin_methods:
        ratio_all = res_all["img_df"].copy()
        ratio_all["ratio"] = res_all["D_img_train"] / res_all["img_df"]["pin_px"].to_numpy(dtype=np.float32)
        exclude = {"NO1"} if apply_1008_quirks else frozenset()
        setup_ratio_all = D.specimen_table_from_images(ratio_all, ratio_all["ratio"].to_numpy(),
                                                        exclude_specimens=exclude)
        predict_fn4, _ = C.fit_linear(setup_ratio_all)

    for ng_sp in D.ng_specimens(row_df):
        ng_mask = (row_df["specimen"] == ng_sp).to_numpy()
        idx_lists_ng, img_df_ng = D.image_table(row_df, ng_mask)
        Xtop_ng = torch.tensor(A.center_crop(cache["top_profiles"][ng_mask]), dtype=torch.float32, device=T.DEVICE)
        Xbot_ng = torch.tensor(A.center_crop(cache["bot_profiles"][ng_mask]), dtype=torch.float32, device=T.DEVICE)
        top_center_ng = torch.tensor(row_df.loc[ng_mask, "top_center"].to_numpy(dtype=np.float32), device=T.DEVICE)
        bot_center_ng = torch.tensor(row_df.loc[ng_mask, "bot_center"].to_numpy(dtype=np.float32), device=T.DEVICE)
        with torch.no_grad():
            top_pred_ng = res_all["net_top"](Xtop_ng)
            bot_pred_ng = res_all["net_bot"](Xbot_ng)
            d_row_ng = (bot_center_ng + bot_pred_ng) - (top_center_ng + top_pred_ng)
            D_img_ng = D.trimmed_mean_rows(d_row_ng, idx_lists_ng, TRIM_FRAC).cpu().numpy()
        pin_img_ng = img_df_ng["pin_px"].to_numpy(dtype=np.float32)
        mm_true_ng = float(img_df_ng["실측mm"].iloc[0])

        direct_pred_ng = float(np.median(convert_mm(D_img_ng, pin_img_ng, v_scale, res_all["c"])))
        판정, 방향 = judge_direction(direct_pred_ng)
        rows.append(dict(variant=tag, specimen=ng_sp, eval_path="model_direct", method="-",
                          mm_true=mm_true_ng, pred_mm=direct_pred_ng,
                          err_um=(direct_pred_ng - mm_true_ng) * 1000.0, 판정=판정, 방향=방향))

        held_specimen_px_ng = float(np.median(D_img_ng))
        for method_name, fitter in (("0_평균찍기", C.fit_const), ("2_전체선형회귀", C.fit_linear),
                                     ("3_물리스케일고정", C.fit_fixed_scale)):
            predict_fn, _ = fitter(setup_px_all)
            pred_mm = predict_fn(held_specimen_px_ng)
            판정, 방향 = judge_direction(pred_mm)
            rows.append(dict(variant=tag, specimen=ng_sp, eval_path="model_px+calib", method=method_name,
                              mm_true=mm_true_ng, pred_mm=pred_mm, err_um=(pred_mm - mm_true_ng) * 1000.0,
                              판정=판정, 방향=방향))

        if predict_fn4 is not None:
            held_ratio_ng = float(np.median(D_img_ng / pin_img_ng))
            pred_mm4 = predict_fn4(held_ratio_ng)
            판정, 방향 = judge_direction(pred_mm4)
            rows.append(dict(variant=tag, specimen=ng_sp, eval_path="model_px+calib", method="4_핀비율정규화선형",
                              mm_true=mm_true_ng, pred_mm=pred_mm4, err_um=(pred_mm4 - mm_true_ng) * 1000.0,
                              판정=판정, 방향=방향))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--only-variant", type=str, default=None,
                     help="디버그용: 'V1_H_l0' 같은 단일 변형 태그만 돈다")
    D.add_io_args(ap)  # --profiles/--meta/--pin-col/--label-source/--cache-dir/--results-dir
    args = ap.parse_args()

    cache, row_df = D.resolve_io_args(args)  # 디렉터리 전환 + 메타 CSV 기준 로드(Codex [high] 반영)
    normal_specimens = D.normal_specimens(row_df)  # 하드코딩 안 함(지시서 8번)

    # Codex adversarial-review [P2] 반영(2026-10-16 2차): pin_px가 메타에 없거나 일부
    # 비어 있으면 V2 환산·방식4(핀비율선형)가 전부 NaN으로 조용히 오염된다 — 그 전에
    # 걸러서 V2 변형을 건너뛰고 방식4도 뺀다. apply_1008_quirks는 묶음1/NO1 제외 규칙이
    # 10-08 데이터(label_source="micrometer") 전용이라는 뜻 — gauge 데이터에서는 우연히
    # 묶음 ID가 1이거나 시편이 "NO1"이어도 제외하지 않는다.
    pin_ok = row_df["pin_px"].notna().all()
    apply_1008_quirks = (args.label_source == "micrometer")
    if not pin_ok:
        print(f"[경고] 메타의 '{args.pin_col}' 열이 없거나 일부 비어 있다 — V2 변형과 "
              "모든 변형의 '4_핀비율정규화선형' 캘리브레이션을 건너뛴다.", flush=True)
    active_variants = [v for v in VARIANTS if pin_ok or v[0] != "V2"]
    active_teacher_scales = [s for s in V_SCALES if pin_ok or s != "V2"]
    if args.only_variant:
        active_tags = {variant_tag(*v) for v in active_variants} | {f"TEACHER_{s}" for s in active_teacher_scales}
        if args.only_variant not in active_tags:
            raise ValueError(f"--only-variant {args.only_variant!r}를 실행할 수 없다 "
                              f"(pin_ok={pin_ok}이라 V2 계열이 빠졌을 수 있다). 가능한 태그: "
                              f"{sorted(active_tags)}")

    full_n = len(row_df)
    oof_store = {}  # variant_tag -> dict(top_pred(full_n,nan), bot_pred(full_n,nan))

    diam_rows = []  # diam_loso.csv
    edge_shift_rows = []  # 참고용 wide 요약은 evaluate.py에서 만든다 — 여기선 원자료
    ng_rows = []

    for v_scale, v_scope, lam in active_variants:
        tag = variant_tag(v_scale, v_scope, lam)
        if args.only_variant and tag != args.only_variant:
            continue
        t_variant0 = time.time()
        oof_top_full = np.full(full_n, np.nan, dtype=np.float32)
        oof_bot_full = np.full(full_n, np.nan, dtype=np.float32)

        for held_out in normal_specimens:
            res = run_fold(held_out, v_scale, v_scope, lam, cache, row_df, epochs=args.epochs,
                            apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)
            diam_rows.append(dict(variant=tag, v_scale=v_scale, v_scope=v_scope, lam=lam,
                                   fold=held_out, eval_path="model_direct", method="-",
                                   **res["direct_row"],
                                   판정=judge_direction(res["direct_row"]["pred_mm"])[0],
                                   방향=judge_direction(res["direct_row"]["pred_mm"])[1]))
            for cr in res["calib_rows"]:
                판정, 방향 = judge_direction(cr["pred_mm"])
                diam_rows.append(dict(variant=tag, v_scale=v_scale, v_scope=v_scope, lam=lam,
                                       fold=held_out, eval_path="model_px+calib", method=cr["method"],
                                       held_out=held_out, mm_true=res["direct_row"]["mm_true"],
                                       pred_mm=cr["pred_mm"], err_um=cr["err_um"], abs_err_um=abs(cr["err_um"]),
                                       판정=판정, 방향=방향))
            edge_shift_rows.append(dict(variant=tag, fold=held_out, **res["edge_shift"]))
            m = res["oof"]["held_mask"]
            oof_top_full[m] = res["oof"]["top_pred"]
            oof_bot_full[m] = res["oof"]["bot_pred"]

        # NG 평가: 정상 10시편 전체로 파인튜닝한 모델(held_out="ALL")
        res_all = run_fold("ALL", v_scale, v_scope, lam, cache, row_df, epochs=args.epochs,
                            apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)
        ng_rows += ng_eval_rows_for(tag, v_scale, res_all, row_df, cache,
                                     apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)

        oof_store[f"{tag}__top"] = oof_top_full
        oof_store[f"{tag}__bot"] = oof_bot_full
        print(f"[변형 {tag}] {round(time.time()-t_variant0,1)}s", flush=True)

    # --- teacher 베이스라인(지시서: "둘째 줄은 teacher(사전학습 모델 그대로, 파인튜닝
    # 없음)") — `run_fold(epochs=0)`로 파인튜닝 루프를 건너뛰면 사전학습 가중치·c0 그대로
    # 같은 평가 코드 경로를 탄다. v_scope/lam은 학습이 없으니 의미 없다(H로 고정). ---
    for v_scale in active_teacher_scales:
        tag = f"TEACHER_{v_scale}"
        if args.only_variant and tag != args.only_variant:
            continue
        for held_out in normal_specimens:
            res = run_fold(held_out, v_scale, "H", 0.0, cache, row_df, epochs=0,
                            apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)
            diam_rows.append(dict(variant=tag, v_scale=v_scale, v_scope="-", lam=np.nan,
                                   fold=held_out, eval_path="model_direct", method="-",
                                   **res["direct_row"],
                                   판정=judge_direction(res["direct_row"]["pred_mm"])[0],
                                   방향=judge_direction(res["direct_row"]["pred_mm"])[1]))
            for cr in res["calib_rows"]:
                판정, 방향 = judge_direction(cr["pred_mm"])
                diam_rows.append(dict(variant=tag, v_scale=v_scale, v_scope="-", lam=np.nan,
                                       fold=held_out, eval_path="model_px+calib", method=cr["method"],
                                       held_out=held_out, mm_true=res["direct_row"]["mm_true"],
                                       pred_mm=cr["pred_mm"], err_um=cr["err_um"], abs_err_um=abs(cr["err_um"]),
                                       판정=판정, 방향=방향))
        res_all = run_fold("ALL", v_scale, "H", 0.0, cache, row_df, epochs=0,
                            apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)
        ng_rows += ng_eval_rows_for(tag, v_scale, res_all, row_df, cache,
                                     apply_1008_quirks=apply_1008_quirks, include_pin_methods=pin_ok)
        print(f"[teacher 베이스라인 {tag}]", flush=True)

    np.savez(os.path.join(D.CACHE_DIR, "finetune_oof.npz"), **oof_store)

    diam_df = pd.DataFrame(diam_rows)
    diam_path = os.path.join(D.RESULTS_DIR, "diam_loso.csv")
    diam_df.to_csv(diam_path, index=False, encoding="utf-8-sig")
    print(f"저장: {diam_path} ({len(diam_df)}행)")

    edge_shift_df = pd.DataFrame(edge_shift_rows)
    edge_shift_raw_path = os.path.join(D.RESULTS_DIR, "edge_shift_raw.csv")
    edge_shift_df.to_csv(edge_shift_raw_path, index=False, encoding="utf-8-sig")
    print(f"저장: {edge_shift_raw_path} ({len(edge_shift_df)}행)")

    # NG 역할 시편이 하나도 없으면(다음 촬영 gauge 데이터처럼, Codex adversarial-review
    # [high] 반영 테스트에서 재현됨) ng_rows가 비어 pd.DataFrame([])가 열 없는 CSV를 써서
    # evaluate.py의 read_csv가 EmptyDataError로 죽는다 — 빈 경우에도 헤더는 쓴다.
    ng_df = pd.DataFrame(ng_rows, columns=NG_COLUMNS)
    ng_path = os.path.join(D.RESULTS_DIR, "ng_eval.csv")
    ng_df.to_csv(ng_path, index=False, encoding="utf-8-sig")
    print(f"저장: {ng_path} ({len(ng_df)}행)")


if __name__ == "__main__":
    main()
