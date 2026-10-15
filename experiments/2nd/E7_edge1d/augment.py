"""E7 라벨-보존 증강.

**입력 계약(Codex adversarial-review [medium] 반영):** 모든 함수가 받는 `profile`은
`build_cache.py`가 저장한 **넓은(89px, `HALF_WINDOW_PX=40` + 양쪽 `SHIFT_BUFFER_PX=4`
버퍼) 프로파일**이고, 반환값은 모델이 실제로 쓰는 **좁은(81px) 크롭**이다. shift만 이
버퍼 안에서 보간하고(±0.5px 이동은 버퍼 4px 안에 넉넉히 들어와 외삽이 필요 없다),
나머지(blur/brightness/contrast/identity)는 버퍼를 무시하고 중앙 81px만 쓴다.
**처음 버전은 이미 81px로 잘라낸 프로파일에 `CubicSpline(..., extrapolate=True)`로
shift를 적용했다** — ±0.5px 이동이면 창 양 끝 중 한쪽은 실제 데이터 범위를 벗어나
스플라인이 "외삽"한 값을 써야 했고, 이게 방향(-0.5px vs +0.5px)마다 반대쪽 끝에서
일어나 보고했던 "-0.5px만 유난히 불안정한" 비대칭의 원인일 수 있다는 지적을 받았다.
지금은 버퍼 안에서만 보간해 외삽이 전혀 없다.

**E2에서 "측정 학습 허용"이 나온 증강만, E2가 실제로 측정한 강도에서만** 쓴다(지시서
E7 3번). 지시서가 적어 둔 증강 어휘(비대칭 블러, 링잉 세기, 내부 반사 레벨, 가우시안
노이즈)는 E2가 아예 테스트하지 않은 종류라 "E2 허용"에 해당하지 않는다 — E2는
(대칭)블러, 밝기, 대비, 감마, 회전, 서브픽셀 평행이동만 테스트했다. 감마는 상·하단 둘
다 "게이트"라 뺐고, 회전은 2D 전체 이미지 기하 변환이라 1D 프로파일에는 직접 적용할
수 없어 뺐다.

**블러 허용 강도 재검토(Codex adversarial-review [medium] 반영):** `E2_aug_check/README.md`
1단계 표가 "전체중앙값"과 "셋업최악" 둘 다 판정한다 — **상단 블러는 E2가 테스트한 세
강도(σ=0.5, 1.0, 2.0) 전부 셋업최악 기준으로는 "게이트"다**(σ=0.5조차 전체중앙값은
허용(0.023px)이지만 최악 셋업은 0.038px로 게이트). 처음 버전은 전체중앙값만 보고
σ∈{0.25,0.5}를 상단에 "허용"으로 넣었는데(σ=0.25는 E2가 아예 테스트하지 않은 값이기도
했다), 둘 다 근거가 부족했다. **지금은 상단 블러를 완전히 뺀다**(`TOP_ALLOWED["blur_sigma"]
= []`). 하단은 σ=0.5·1.0이 전체중앙값·셋업최악 둘 다 "허용"이라 그대로 쓴다(σ=2.0은
둘 다 게이트라 계속 뺌, σ=0.25는 미검증이라 뺌).
- blur: 상단 **없음**(E2가 테스트한 모든 강도가 셋업최악 기준 게이트), 하단
  σ∈{0.5,1.0}(전체중앙값·셋업최악 둘 다 허용인 값만).
- brightness: 상단 {-10,-20}, 하단 {-10,-20,-40}(어둡게만 — 밝게는 포화로 전부 게이트).
- contrast: ×0.8(둘 다 허용, Δ=0 정확히).
- subpixel shift: 라벨이 정확히 함께 이동하는 유일한 증강(양성 대조군, E2 3단계).

라벨 불변 증강(blur/brightness/contrast)은 "E2가 측정한 Δ가 0.03px 판정선 안쪽"이라는
근거로 라벨을 그대로 둔다 — 완전히 0이라는 보장은 아니다(E2 한계와 동일).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

HALF_WINDOW_PX = 40  # build_cache.py와 같은 값(모델이 보는 폭 = 2*40+1 = 81)
SHIFT_BUFFER_PX = 4  # build_cache.py와 같은 값(저장된 프로파일 폭 = 81 + 2*4 = 89)
MODEL_WIDTH = 2 * HALF_WINDOW_PX + 1

TOP_ALLOWED = {
    "blur_sigma": [],
    "brightness_delta": [-10.0, -20.0],
    "contrast_factor": [0.8],
}
BOTTOM_ALLOWED = {
    "blur_sigma": [0.5, 1.0],
    "brightness_delta": [-10.0, -20.0, -40.0],
    "contrast_factor": [0.8],
}
SHIFT_RANGE_PX = (-0.5, 0.5)  # E2 3단계가 검증한 범위(0.25, 0.5px)를 포함하는 연속 구간


def allowed_for(edge: str) -> dict:
    if edge == "top":
        return TOP_ALLOWED
    if edge == "bottom":
        return BOTTOM_ALLOWED
    raise ValueError(f"unknown edge: {edge!r}")


def center_crop(profile_wide: np.ndarray) -> np.ndarray:
    """(...,89) -> (...,81) 중앙 크롭. shift 이외의 모든 증강이 이걸 먼저 쓴다."""
    return profile_wide[..., SHIFT_BUFFER_PX:SHIFT_BUFFER_PX + MODEL_WIDTH]


def shift1d(profile_wide: np.ndarray, dy: float) -> np.ndarray:
    """서브픽셀 평행이동. 넓은(89) 버퍼 안에서만 보간해 외삽이 없다(위 모듈 docstring).
    new_profile(i) = old_profile_wide(SHIFT_BUFFER_PX + i - dy) for i in [0, 80].
    라벨은 `label + dy`."""
    x_wide = np.arange(len(profile_wide), dtype=np.float64)
    cs = CubicSpline(x_wide, profile_wide, extrapolate=True)  # 버퍼 설계상 실제로는 항상 범위 안
    out_idx = np.arange(MODEL_WIDTH, dtype=np.float64)
    src = SHIFT_BUFFER_PX + out_idx - dy
    return cs(src)


def blur1d(profile_wide: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(center_crop(profile_wide), sigma=sigma, mode="nearest")


def brightness1d(profile_wide: np.ndarray, delta: float) -> np.ndarray:
    return center_crop(profile_wide) + delta


def contrast1d(profile_wide: np.ndarray, factor: float) -> np.ndarray:
    return (center_crop(profile_wide) - 128.0) * factor + 128.0


def random_augment(profile_wide: np.ndarray, label: float, edge: str, rng: np.random.Generator):
    """profile_wide(1개, 길이 89)·label(스칼라)에 E2-허용 증강 중 하나를 무작위로 적용.
    "identity"도 후보에 포함해 일부 샘플은 그대로 둔다(중앙 81px 크롭만). 상단은 허용된
    블러 강도가 없어 "blur"가 애초에 후보에서 빠진다(그래서 상단 선택지는 4개, 하단은
    5개 — 균등 확률이 아니라는 뜻, 의도된 비대칭이다). (profile_out, label_out) 반환."""
    allowed = allowed_for(edge)
    choices = ["identity", "shift", "brightness", "contrast"]
    if allowed["blur_sigma"]:
        choices.append("blur")
    kind = rng.choice(choices)
    if kind == "identity":
        return center_crop(profile_wide).copy(), label
    if kind == "shift":
        dy = rng.uniform(*SHIFT_RANGE_PX)
        return shift1d(profile_wide, dy), label + dy
    if kind == "blur":
        sigma = rng.choice(allowed["blur_sigma"])
        return blur1d(profile_wide, sigma), label
    if kind == "brightness":
        delta = rng.choice(allowed["brightness_delta"])
        return brightness1d(profile_wide, delta), label
    if kind == "contrast":
        factor = rng.choice(allowed["contrast_factor"])
        return contrast1d(profile_wide, factor), label
    raise AssertionError(kind)


def augment_batch(profiles_wide: np.ndarray, labels: np.ndarray, edge: str, rng: np.random.Generator):
    """(B,89)·(B,) 배치에 샘플마다 독립적으로 `random_augment`를 적용 -> (B,81)·(B,)."""
    out_p = np.empty((len(profiles_wide), MODEL_WIDTH), dtype=profiles_wide.dtype)
    out_l = np.empty_like(labels)
    for i in range(len(profiles_wide)):
        out_p[i], out_l[i] = random_augment(profiles_wide[i], float(labels[i]), edge, rng)
    return out_p, out_l


# 평가용 고정 강도 증강(강건성 표, train_eval.py에서 사용) — 이름: (종류, 강도)
# 상단은 blur 항목이 없다 — E2가 테스트한 모든 강도가 셋업최악 기준 게이트라 "라벨
# 불변"이라는 전제 자체가 top에는 근거가 없다(위 모듈 docstring).
EVAL_INTENSITIES = {
    "top": [
        ("clean", None),
        ("shift", 0.25), ("shift", 0.5), ("shift", -0.5),
        ("brightness", -10.0), ("brightness", -20.0),
        ("contrast", 0.8),
    ],
    "bottom": [
        ("clean", None),
        ("shift", 0.25), ("shift", 0.5), ("shift", -0.5),
        ("blur", 0.5), ("blur", 1.0),
        ("brightness", -10.0), ("brightness", -20.0), ("brightness", -40.0),
        ("contrast", 0.8),
    ],
}


def apply_eval_augment(profiles_wide: np.ndarray, labels: np.ndarray, kind: str, intensity):
    """(B,89) 넓은 프로파일 + teacher 라벨 -> ((B,81) 증강된 프로파일, 보정된 라벨)."""
    if kind == "clean":
        return center_crop(profiles_wide).copy(), labels.copy()
    if kind == "shift":
        out_p = np.stack([shift1d(p, intensity) for p in profiles_wide])
        return out_p, labels + intensity
    if kind == "blur":
        out_p = np.stack([blur1d(p, intensity) for p in profiles_wide])
        return out_p, labels.copy()
    if kind == "brightness":
        return center_crop(profiles_wide) + intensity, labels.copy()
    if kind == "contrast":
        return (center_crop(profiles_wide) - 128.0) * intensity + 128.0, labels.copy()
    raise ValueError(kind)
