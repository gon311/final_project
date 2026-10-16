"""E10 공용 데이터 로딩: E7 `profiles.npz`(열별 상·하단 프로파일 + teacher 라벨)에
메타 CSV(묶음·시편·역할·실측mm·핀px)를 파일명으로 합친다. E7·E6 코드는 import만 하고
수정하지 않는다(지시서).

**다음 촬영 대비 일반화(지시서 8번, Codex adversarial-review 2026-10-16 반영):**
- `load_merged_cache`가 `profiles_path`/`meta_path`/`pin_col`/`label_source`를 인자로
  받는다. 모든 스크립트의 `main()`이 이 네 인자를 CLI 옵션(`add_io_args`/`resolve_io_args`
  참고)으로 받아 그대로 전달한다 — **이전 버전은 저수준 함수만 인자를 받고 `main()`은
  기본 경로만 썼다(Codex [high] 지적, 실제로는 CLI로 새 촬영 데이터를 가리킬 방법이
  없었다).**
- **묶음·시편·역할은 이제 메타 CSV에서만 읽는다(캐시 npz의 동명 필드는 쓰지 않는다)**
  — 이전 버전은 메타에서 `실측mm`만 가져오고 묶음·시편·역할은 캐시(`cache["bundle"]`
  등)에서 읽었는데, 지시서 8번이 "필수 열: 파일, 묶음, 시편, 역할, 실측mm"라고 명시한
  건 메타가 분류 정보의 **단일 출처**라는 뜻이다(Codex [high] 지적 — 캐시에 의존하면
  다음 촬영에서 분류표의 `역할=gauge`·물체 ID가 전혀 반영되지 않는 경로가 생긴다).
  `label_source="gauge"`면 메타의 `실측mm` 열이 이미 공칭 지름을 담고 있다고 보고 그대로
  쓰고, LOSO 그룹(`normal_specimens`)은 메타의 `시편`을 그대로 쓴다 — 핀게이지는 지름별로
  다른 `시편` 값(예: `GAUGE_20.00`)을 쓰면 자동으로 지름 단위 LOSO가 된다.
- **프로파일 폭은 npz 자체가 적어 둔 `half_window_px`/`shift_buffer_px`로 검증한다**(아래
  `_load_profiles_npz`). 다만 이 값이 E7 `augment.py`의 고정 상수(`HALF_WINDOW_PX=40`,
  `SHIFT_BUFFER_PX=4`)와 다르면 **E10은 그 npz를 처리할 수 없다** — `A.center_crop`·
  `HeatmapHead`의 중심 정렬이 이 상수에 고정돼 있고, E10은 E7 파일을 수정하지 않기로
  했기 때문이다(Codex [medium] 지적 — 이전 버전은 "89px가 아니면 거부"라고만 하고 왜
  거부하는지, 무엇을 바꿔야 하는지 말하지 않았다). 다음 촬영이 다른 물리적 창 폭을
  쓰려면 **E7 `augment.py`/`build_cache.py`의 상수부터 바꿔야 한다** — E10 한계로
  README에 명시한다.
- 10-08 데이터에는 gauge 행이 없으므로 이 경로는 `synthetic_path_test.py`의 합성
  npz·CSV로, **그리고 `pretrain.py`/`finetune.py`를 실제 CLI(서브프로세스)로 호출해서**
  테스트한다(지시서: "합성 수치로 결론 내지 않는다" — 통과/실패만 본다).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO2ND = os.path.join(HERE, "..")
sys.path.insert(0, REPO2ND)
sys.path.insert(0, os.path.join(REPO2ND, "E7_edge1d"))
sys.path.insert(0, os.path.join(REPO2ND, "E6_calibration"))

import augment as A  # noqa: E402  (E7, import만 — 수정 안 함)
import model as M  # noqa: E402  (E7)
import train_eval as T  # noqa: E402  (E7 — load_cache 검증, train_net, predict, DEVICE 재사용)
import calibration as C  # noqa: E402  (E6 — fit_const/fit_linear/fit_fixed_scale 등 재사용)

DEFAULT_PROFILES_NPZ = os.path.join(REPO2ND, "cache", "E7_edge1d", "profiles.npz")
DEFAULT_META_CSV = os.path.join(REPO2ND, "E6_calibration", "results", "values_by_image.csv")

CACHE_DIR = os.path.join(REPO2ND, "cache", "E10_diam_finetune")
RESULTS_DIR = os.path.join(HERE, "results")

PHYSICAL_SCALE_MM_PER_PX = C.PHYSICAL_SCALE  # 0.0367, E6와 같은 상수(0.2절)

META_COLUMN_ALIASES = {"파일": "filename", "file": "filename",
                        "묶음": "bundle", "시편": "specimen", "역할": "role"}
REQUIRED_META_COLUMNS = ("filename", "bundle", "specimen", "role", "실측mm")


def set_run_dirs(cache_dir: str = None, results_dir: str = None):
    """체크포인트·결과 저장 위치를 바꾼다(다음 촬영 재실행 시 10-08 산출물과 섞이지
    않게, Codex adversarial-review [high] 반영 — CLI에서 받은 값을 여기로 흘린다)."""
    global CACHE_DIR, RESULTS_DIR
    if cache_dir:
        CACHE_DIR = cache_dir
    if results_dir:
        RESULTS_DIR = results_dir
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def add_io_args(parser):
    """`--profiles`/`--meta`/`--pin-col`/`--label-source`/`--cache-dir`/`--results-dir`를
    `argparse.ArgumentParser`에 등록한다 — `pretrain.py`/`finetune.py`/`step_diag.py`가
    공유한다(Codex [high] 반영: 이전에는 이 인자들이 저수준 함수에만 있고 CLI에는
    없었다)."""
    parser.add_argument("--profiles", type=str, default=None,
                         help=f"profiles.npz 경로(기본: {DEFAULT_PROFILES_NPZ})")
    parser.add_argument("--meta", type=str, default=None,
                         help=f"메타 CSV 경로, 필수 열 {REQUIRED_META_COLUMNS}(기본: {DEFAULT_META_CSV})")
    parser.add_argument("--pin-col", type=str, default="pin_cross50_px",
                         help="메타에서 핀 px로 쓸 열 이름(없으면 V2 변형은 NaN으로 평가 제외)")
    parser.add_argument("--label-source", type=str, default="micrometer",
                         choices=["micrometer", "gauge"],
                         help="micrometer=샤프트 실측, gauge=공칭 핀게이지 지름(다음 촬영 GT)")
    parser.add_argument("--cache-dir", type=str, default=None,
                         help="체크포인트·중간 캐시 저장 위치(기본: cache/E10_diam_finetune)")
    parser.add_argument("--results-dir", type=str, default=None,
                         help="결과 CSV 저장 위치(기본: E10_diam_finetune/results)")


def resolve_io_args(args):
    """`add_io_args`로 등록한 인자를 적용한다: 디렉터리를 전환하고 `(cache, row_df)`를
    돌려준다."""
    set_run_dirs(args.cache_dir, args.results_dir)
    return load_merged_cache(args.profiles, args.meta, args.pin_col, args.label_source)


def normal_specimens(row_df: pd.DataFrame) -> list:
    """LOSO 그룹(= "물체 ID") 목록을 **데이터에서 직접** 뽑는다(지시서 8번: "LOSO 그룹은
    물체 ID로 한다") — `specimen` 값을 하드코딩하지 않는다. 10-08 데이터에서는 결과가
    `["NO1",...,"NO10"]`과 같고, 공칭 핀게이지 분류표(`label_source="gauge"`)에서는
    지름별 `specimen`(예: `GAUGE_20.00`)이 그대로 LOSO 그룹이 된다."""
    return sorted(row_df.loc[row_df["role"] != "ng", "specimen"].unique().tolist())


def ng_specimens(row_df: pd.DataFrame) -> list:
    return sorted(row_df.loc[row_df["role"] == "ng", "specimen"].unique().tolist())


def _load_profiles_npz(path: str) -> dict:
    """프로파일 폭을 npz 자체의 half_window_px/shift_buffer_px 메타데이터로 검증한다
    (Codex adversarial-review [medium] 반영, 2026-10-16) - E7 상수(A.HALF_WINDOW_PX,
    A.SHIFT_BUFFER_PX)와 고정 매직넘버(89) 하나로만 비교하던 이전 버전보다 구체적인
    오류를 낸다: 실제 배열 폭이 npz가 적어 둔 half_window·buffer와 안 맞으면(메타데이터
    자체가 틀렸거나 손상됐다는 뜻) 그렇게 알리고, npz 메타데이터는 자기 일관적인데 E7
    상수와 다르면(= 다른 물리적 창 폭으로 만든 next-shoot 캐시라는 뜻) E10이 아니라
    E7 augment.py/build_cache.py의 상수부터 바꿔야 한다고 알린다(A.center_crop과
    HeatmapHead의 중심 정렬이 그 상수에 고정돼 있고, E10은 E7 파일을 수정하지 않는다)."""
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    for meta_key in ("half_window_px", "shift_buffer_px"):
        if meta_key not in out:
            raise RuntimeError(f"{path}에 '{meta_key}' 메타데이터가 없다 - build_cache.py가 "
                                "저장하는 필드다. 이 필드 없이는 프로파일 폭을 검증할 수 없다.")
    half_window = int(out["half_window_px"][0])
    shift_buffer = int(out["shift_buffer_px"][0])
    expected_width = 2 * (half_window + shift_buffer) + 1
    for key in ("top_profiles", "bot_profiles"):
        width = out[key].shape[-1]
        if width != expected_width:
            raise RuntimeError(
                f"{path}의 '{key}' 폭이 {width}px인데 저장된 half_window_px={half_window}, "
                f"shift_buffer_px={shift_buffer}로 계산한 기대 폭은 {expected_width}px다 - "
                "npz 메타데이터와 실제 배열이 어긋난, 손상되었거나 다른 스크립트가 만든 캐시다."
            )
    if half_window != A.HALF_WINDOW_PX or shift_buffer != A.SHIFT_BUFFER_PX:
        raise RuntimeError(
            f"{path}는 half_window_px={half_window}, shift_buffer_px={shift_buffer}로 만들어졌는데 "
            f"E7 augment.py의 고정 상수는 HALF_WINDOW_PX={A.HALF_WINDOW_PX}, "
            f"SHIFT_BUFFER_PX={A.SHIFT_BUFFER_PX}다. A.center_crop과 HeatmapHead의 중심 정렬이 "
            "이 상수에 고정돼 있어 E10 코드만으로는 다른 창 폭을 처리할 수 없다 - 다음 촬영이 "
            "다른 물리적 창 폭을 쓰려면 E7 augment.py/build_cache.py의 상수부터 바꿔야 한다"
            "(E10은 E7 파일을 수정하지 않는다, README 한계 참고)."
        )
    return out


def load_merged_cache(profiles_path: str = None, meta_path: str = None,
                       pin_col: str = "pin_cross50_px", label_source: str = "micrometer"):
    """반환: (cache_dict, row_df). 묶음/시편/역할/실측mm은 전부 메타 CSV에서 읽는다
    (캐시 npz의 동명 필드는 쓰지 않는다) - 메타가 분류 정보의 단일 출처다(지시서 8번,
    Codex adversarial-review [high] 반영, 2026-10-16: 이전 버전은 캐시의 bundle/specimen/
    role을 그대로 썼는데, 그러면 메타의 역할=gauge, 물체 ID가 다음 촬영 경로에서 전혀
    반영되지 않았다). label_source="gauge"면 메타의 실측mm 열이 이미 공칭 지름을 담고
    있다고 보고 그대로 쓴다. 파일명이 메타에 없는 행, 메타에 파일명이 중복된 행은 바로
    에러를 낸다(조용히 NaN이나 행 폭발로 넘어가지 않는다)."""
    profiles_path = profiles_path or DEFAULT_PROFILES_NPZ
    meta_path = meta_path or DEFAULT_META_CSV
    cache = _load_profiles_npz(profiles_path)

    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    meta = meta.rename(columns={k: v for k, v in META_COLUMN_ALIASES.items() if k in meta.columns})
    missing = [c for c in REQUIRED_META_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"{meta_path}에 필수 열이 없다: {missing} (있는 열: {list(meta.columns)})")
    if meta["filename"].duplicated().any():
        dup = meta.loc[meta["filename"].duplicated(), "filename"].unique().tolist()
        raise ValueError(f"{meta_path}에 파일명이 중복됐다: {dup[:5]}{'...' if len(dup) > 5 else ''}")

    keep = list(REQUIRED_META_COLUMNS) + ([pin_col] if pin_col and pin_col in meta.columns else [])
    meta = meta[keep]

    row_df = pd.DataFrame(dict(
        filename=cache["filename"], x=cache["x"],
        top_center=cache["top_center"].astype(np.float64),
        bot_center=cache["bot_center"].astype(np.float64),
        top_label=cache["top_label"].astype(np.float64),
        bot_label=cache["bot_label"].astype(np.float64),
    ))
    n_before = len(row_df)
    row_df = row_df.merge(meta, on="filename", how="left", indicator=True)
    unmatched = row_df.loc[row_df["_merge"] != "both", "filename"].unique().tolist()
    if unmatched:
        raise ValueError(f"{len(unmatched)}개 파일명이 {meta_path}에 없다(캐시에만 있음): "
                          f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")
    assert len(row_df) == n_before, "메타 조인으로 행 수가 바뀌었다(중복 검사를 통과했는데도) - 버그"
    row_df = row_df.drop(columns=["_merge"])

    if pin_col in row_df.columns:
        row_df = row_df.rename(columns={pin_col: "pin_px"})
    else:
        row_df["pin_px"] = np.nan
    # pin_px가 없거나 일부만 있어도 여기서는 에러를 내지 않는다 - 핀 없이도(V1만) 돌려야
    # 하는 정상적인 경로이기 때문이다. 대신 pin_px로 나누는 모든 연산(V2 환산, 방식4
    # 핀비율선형)의 호출부(`finetune.py main()`)가 `row_df["pin_px"].notna().all()`을
    # 확인해 명시적으로 건너뛸 책임을 진다(Codex adversarial-review [P2] 반영,
    # 2026-10-16 2차 - 이전에는 이 NaN이 조용히 계산에 섞여 들어갔다).
    if label_source not in ("micrometer", "gauge"):
        raise ValueError(f"unknown label_source: {label_source!r}")
    return cache, row_df


def trimmed_mean_rows(values, idx_lists, trim_frac: float = 0.1):
    """이미지별 절사평균(상·하위 10%). `values`는 torch.Tensor(미분 가능), `idx_lists`는
    `image_table`이 돌려준 지역 인덱스 배열 리스트다. torch.Tensor를 반환(그래프 유지)."""
    import torch
    out = []
    for idx in idx_lists:
        v = values[idx]
        k = v.shape[0]
        if k >= 5:
            lo = int(np.floor(k * trim_frac))
            hi = k - lo
            v_sorted, _ = torch.sort(v)
            v = v_sorted[lo:hi]
        out.append(v.mean())
    return torch.stack(out)


def specimen_image_table(filenames, specimens_per_image) -> pd.DataFrame:
    """이미지 리스트 -> 이미지 순서를 보존한 (filename, specimen) 표. 호출부가
    이미지 단위 D_i를 시편 단위로 다시 묶을 때 쓴다."""
    return pd.DataFrame(dict(filename=filenames, specimen=specimens_per_image))


def image_table(row_df: pd.DataFrame, mask: np.ndarray):
    """`image_groups`와 같은 그룹화를 한 번만 수행해 (idx_lists, img_df)를 함께 돌려준다
    — `img_df`는 `idx_lists`와 같은 순서로 이미지당 1행(filename·specimen·bundle·실측mm·
    pin_px, 모두 그 이미지의 모든 열이 공유하는 상수값)이다. finetune.py가 이미지 단위
    라벨(실측mm)·핀px·시편(나중에 fold 안 캘리브레이션을 다시 맞출 때)을 D_i와 같은
    순서로 쓰기 위한 것 — 두 번 그룹화해서 순서가 어긋나는 실수를 막는다."""
    sub = row_df.loc[mask].reset_index(drop=True)
    groups = sub.groupby("filename", sort=False)
    idx_lists = [np.asarray(idx, dtype=np.int64) for idx in groups.indices.values()]
    img_df = groups.agg(specimen=("specimen", "first"), bundle=("bundle", "first"),
                         실측mm=("실측mm", "first"), pin_px=("pin_px", "first")).reset_index()
    return idx_lists, img_df


def specimen_table_from_images(img_df: pd.DataFrame, D_img: np.ndarray, exclude_specimens=frozenset()) -> pd.DataFrame:
    """이미지별 D_i(모델이 만든 지름, px 또는 비율)를 시편 단위로 다시 묶는다
    (E6 `calibration.setup_table`과 같은 집계 방식: 시편 내 여러 이미지는 median).
    `exclude_specimens`에 있는 시편은 표에서 뺀다(E6 핀비율 방식의 NO1 제외와 같은 용도)."""
    d = img_df.copy()
    d["px"] = D_img
    d = d[~d["specimen"].isin(exclude_specimens)]
    agg = d.groupby("specimen").agg(실측mm=("실측mm", "first"), px=("px", "median"))
    return agg
