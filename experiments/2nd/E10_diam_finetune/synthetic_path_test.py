"""E10 8단계: 다음 촬영(공칭 핀게이지 GT) 경로가 코드로 끝까지 도는지만 확인하는
합성 테스트. **합성 수치로 어떤 결론도 내지 않는다**(지시서) — 이 스크립트는 pass/fail만
본다.

**Codex adversarial-review [high] 반영(2026-10-16, 2차):** 이전 버전은 `pretrain.py`/
`finetune.py`의 `main()`을 건너뛰고 저수준 함수(`train_net`, `run_fold`)를 직접 호출했다
— CLI 자체가 `--profiles`/`--meta`/`--pin-col`/`--label-source`를 받지 않는다는 결함을
가렸다. **지금은 다섯 스크립트를 전부 실제 서브프로세스 CLI로 돌린다**
(`pretrain.py` → `finetune.py` → `evaluate.py` → `step_diag.py` → `make_figures.py`,
각각 `python3 <script>.py --profiles ... --meta ... --cache-dir ... --results-dir ...`).

메타 CSV는 `묶음`/`시편`/`역할`/`실측mm`(+ 핀 열)을 전부 담는다(`data.py`가 이제 이
네 열을 메타에서만 읽는다, Codex [high] 반영 — 캐시 npz의 동명 필드가 있어도 쓰지
않는다는 걸 이 테스트가 확인한다). LOSO 그룹은 `data.normal_specimens`가 **메타의
`시편`에서** 뽑으므로 공칭 지름별 `시편`(`GAUGE_10.00` 등)이 그대로 LOSO 단위이 된다.

**Codex adversarial-review [P2] 반영(2026-10-16, 3차):** 이전 버전은 핀 열이 **있는**
메타로만 테스트해서 "핀 열이 없으면 NaN이 결과에 섞여 들어간다"는 지적을 못 잡았다.
또 V2(핀비율) 변형을 돌린 적이 없어서 "묶음1 제외가 gauge 데이터에도 무조건 적용된다"는
지적도 못 잡았다. 지금은 두 시나리오를 추가한다: (6) 핀 열 없는 메타로 `finetune.py`를
돌려 V2·방식4가 **깨지지 않고 아예 빠지는지** 확인, (7) `묶음=1`인 gauge 이미지가
`apply_1008_quirks=False`에서는 V2 학습에 들어가고 `True`에서는 빠지는지 `run_fold`를
직접 불러 학습 이미지 집합을 비교한다.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RNG = np.random.default_rng(0)
WIDTH = 89
N_SPECIMENS = 3
N_IMAGES_PER_SPECIMEN = 3
N_COLS_PER_IMAGE = 6
NOMINAL_MM = {0: 10.00, 1: 15.00, 2: 20.00}


def build_synthetic(tmp_dir: str):
    rows_top, rows_bot = [], []
    labels_top, labels_bot = [], []
    top_center, bot_center = [], []
    x_all, filename_all = [], []

    meta_rows = []
    for sp_idx in range(N_SPECIMENS):
        specimen = f"GAUGE_{NOMINAL_MM[sp_idx]:.2f}"
        for img_idx in range(N_IMAGES_PER_SPECIMEN):
            fn = f"synthetic_{sp_idx}_{img_idx}.jpg"
            meta_rows.append(dict(파일=fn, 묶음=sp_idx * 10 + img_idx, 시편=specimen, 역할="gauge",
                                   실측mm=NOMINAL_MM[sp_idx], pin_cross50_px=300.0))
            for col in range(N_COLS_PER_IMAGE):
                base_top = RNG.normal(0, 3, WIDTH).cumsum() + 100
                base_bot = RNG.normal(0, 3, WIDTH).cumsum() + 100
                rows_top.append(base_top.astype(np.float32))
                rows_bot.append(base_bot.astype(np.float32))
                labels_top.append(float(RNG.uniform(-3, 0)))
                labels_bot.append(float(RNG.uniform(6, 12)))
                top_center.append(100)
                bot_center.append(100 + int(NOMINAL_MM[sp_idx] / 0.0367))  # 대략적인 px 지름
                x_all.append(col * 2)
                filename_all.append(fn)

    npz_path = os.path.join(tmp_dir, "synthetic_profiles.npz")
    np.savez(npz_path,
              top_profiles=np.stack(rows_top), bot_profiles=np.stack(rows_bot),
              top_label=np.array(labels_top, dtype=np.float32), bot_label=np.array(labels_bot, dtype=np.float32),
              top_center=np.array(top_center, dtype=np.int32), bot_center=np.array(bot_center, dtype=np.int32),
              x=np.array(x_all, dtype=np.int32), filename=np.array(filename_all),
              half_window_px=np.array([40]), shift_buffer_px=np.array([4]), col_step_px=np.array([2]))
    # 캐시 npz에도 bundle/specimen/role을 일부러 **다르게** 넣어 둔다 — data.py가 정말
    # 메타만 읽고 캐시 값은 무시하는지(Codex [high] 반영) 이 테스트 자체로 검증하기 위함.
    bogus = dict(np.load(npz_path, allow_pickle=False))
    bogus["bundle"] = np.zeros(len(filename_all), dtype=np.int32)
    bogus["specimen"] = np.array(["BOGUS"] * len(filename_all))
    bogus["role"] = np.array(["ng"] * len(filename_all))  # 전부 ng면 메타를 안 쓸 때 즉시 깨진다
    np.savez(npz_path, **bogus)

    meta_path = os.path.join(tmp_dir, "synthetic_meta.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
    return npz_path, meta_path


def meta_without_pin(meta_path: str, tmp_dir: str) -> str:
    """핀 열이 아예 없는 메타 CSV를 만든다(Codex [P2] 1차 시나리오)."""
    df = pd.read_csv(meta_path, encoding="utf-8-sig").drop(columns=["pin_cross50_px"])
    out_path = os.path.join(tmp_dir, "synthetic_meta_nopin.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run(cmd, label):
    print(f"  $ {' '.join(os.path.basename(c) if c == cmd[1] else c for c in cmd)}")
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"{label} 실패(exit {r.returncode})")
    return r


def main():
    tmp_dir = tempfile.mkdtemp(prefix="e10_synth_")
    try:
        npz_path, meta_path = build_synthetic(tmp_dir)
        cache_dir = os.path.join(tmp_dir, "cache")
        results_dir = os.path.join(tmp_dir, "results")
        figures_dir = os.path.join(tmp_dir, "figures")
        py = sys.executable
        io_args = ["--profiles", npz_path, "--meta", meta_path, "--pin-col", "pin_cross50_px",
                   "--label-source", "gauge", "--cache-dir", cache_dir, "--results-dir", results_dir]

        run([py, "pretrain.py", "--epochs", "2"] + io_args, "pretrain.py")
        n_ckpt = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
        assert n_ckpt == (N_SPECIMENS + 1) * 2, f"체크포인트 개수가 예상과 다르다: {n_ckpt}"
        print(f"[1/7] pretrain.py --label-source gauge CLI OK ({n_ckpt}개 체크포인트, "
              f"LOSO 그룹을 메타의 '시편'에서 뽑음 — 캐시의 bogus bundle/specimen/role은 무시됨)")

        run([py, "finetune.py", "--epochs", "2", "--only-variant", "V1_H_l0"] + io_args, "finetune.py")
        diam_path = os.path.join(results_dir, "diam_loso.csv")
        assert os.path.exists(diam_path)
        diam = pd.read_csv(diam_path)
        assert set(diam["variant"].unique()) == {"V1_H_l0"}
        print(f"[2/7] finetune.py CLI OK ({len(diam)}행, {diam_path})")

        run([py, "evaluate.py", "--results-dir", results_dir], "evaluate.py")
        assert os.path.exists(os.path.join(results_dir, "summary.csv"))
        print("[3/7] evaluate.py CLI OK")

        run([py, "step_diag.py"] + io_args, "step_diag.py")
        assert os.path.exists(os.path.join(results_dir, "step_occupancy.csv"))
        print("[4/7] step_diag.py CLI OK")

        run([py, "make_figures.py", "--results-dir", results_dir, "--figures-dir", figures_dir],
            "make_figures.py")
        pngs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]
        assert len(pngs) == 3, pngs
        print(f"[5/7] make_figures.py CLI OK ({pngs})")

        # --- (6) 핀 열이 아예 없는 메타: V2·방식4가 깨지지 않고 빠지는지(Codex [P2]) ---
        nopin_meta = meta_without_pin(meta_path, tmp_dir)
        cache_dir_nopin = os.path.join(tmp_dir, "cache_nopin")
        results_dir_nopin = os.path.join(tmp_dir, "results_nopin")
        io_args_nopin = ["--profiles", npz_path, "--meta", nopin_meta, "--pin-col", "pin_cross50_px",
                          "--label-source", "gauge", "--cache-dir", cache_dir_nopin,
                          "--results-dir", results_dir_nopin]
        run([py, "pretrain.py", "--epochs", "2"] + io_args_nopin, "pretrain.py(핀 없음)")
        run([py, "finetune.py", "--epochs", "2"] + io_args_nopin, "finetune.py(핀 없음)")
        diam_nopin = pd.read_csv(os.path.join(results_dir_nopin, "diam_loso.csv"))
        v_scales_seen = diam_nopin["v_scale"].unique().tolist()
        assert "V2" not in v_scales_seen, f"핀 없는데 V2가 돌았다: {v_scales_seen}"
        assert not (diam_nopin["method"] == "4_핀비율정규화선형").any(), "핀 없는데 방식4가 있다"
        assert not diam_nopin["pred_mm"].isna().any(), "pred_mm에 NaN이 남아 있다"
        print(f"[6/7] 핀 열 없는 메타 CLI OK — V2·방식4 없이 V1 변형만 돈다"
              f"(v_scale={sorted(v_scales_seen)}, NaN 없음)")

        # --- (7) 묶음=1인 gauge 이미지가 apply_1008_quirks에 따라 V2 학습에 포함/제외되는지 ---
        import data as D
        import finetune as F
        D.set_run_dirs(cache_dir=cache_dir, results_dir=results_dir)  # (1)~(5)가 쓴 체크포인트 재사용
        cache, row_df = D.load_merged_cache(npz_path, meta_path, "pin_cross50_px", "gauge")
        bundle1_file = "synthetic_0_1.jpg"  # GAUGE_10.00의 묶음=1 이미지(build_synthetic 참고)
        assert (row_df.loc[row_df["filename"] == bundle1_file, "bundle"] == 1).all()

        # held_out="ALL"만 img_df를 돌려준다(run_fold 반환값 계약) — 정상 전체 학습 집합에
        # 묶음1 이미지가 포함/제외되는지로 같은 분기를 검증한다.
        res_strict = F.run_fold("ALL", "V2", "H", 0.0, cache, row_df, epochs=1,
                                 apply_1008_quirks=True, include_pin_methods=True)
        res_loose = F.run_fold("ALL", "V2", "H", 0.0, cache, row_df, epochs=1,
                                apply_1008_quirks=False, include_pin_methods=True)
        in_strict = bundle1_file in set(res_strict["img_df"]["filename"])
        in_loose = bundle1_file in set(res_loose["img_df"]["filename"])
        assert not in_strict, "apply_1008_quirks=True인데 묶음1 이미지가 V2 학습에 들어갔다"
        assert in_loose, "apply_1008_quirks=False(gauge)인데도 묶음1 이미지가 V2 학습에서 빠졌다"
        print(f"[7/7] 묶음1 제외 분기 OK — apply_1008_quirks=True: 제외됨, False(gauge): 포함됨")

        print()
        print("합성 경로 테스트(실제 CLI 5개 + 핀 결측·묶음1 분기 2개) 전부 통과 — 수치는")
        print("무작위 합성 데이터이므로 어떤 결론도 내지 않는다(지시서). 다음 촬영 GT가 오면")
        print("이 테스트가 쓴 것과 같은 --profiles/--meta/--pin-col/--label-source gauge")
        print("인자로 실제 npz·CSV를 가리키면 된다.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
