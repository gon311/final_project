"""E6 불확실성 참고치(Codex adversarial-review 반영): 재장착 반복이 없으니(규칙 8) 적어도
**같은 장착 안 10장 반복 프레임의 흔들림**을 숫자로 내서, "방식 간 MAE 차이가 이 흔들림보다
큰지 작은지" 비교할 잣대로 쓴다. 이건 재장착 변동의 대리값이 아니다 — 같은 장착이라 아마도
재장착 변동보다 작을 것이다(즉 아래 숫자조차 방식 비교의 "하한"이지 실제 불확실성이 아니다).

입력: results/values_by_image.csv
산출물: results/within_mount_repeatability.csv
"""
import os

import pandas as pd

from calibration import PHYSICAL_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
DEFS = {"cross50": "bearing_cross50_px", "grad_peak": "bearing_grad_peak_px", "moment": "bearing_moment_px"}


def main():
    df = pd.read_csv(os.path.join(HERE, "results", "values_by_image.csv"), encoding="utf-8-sig")
    df = df[df["묶음"] <= 10]  # 정상 10개 셋업만(참고치라 NG까지 넣을 필요 없음)

    rows = []
    for name, col in DEFS.items():
        setup_sigma_px = df.groupby("묶음")[col].std()
        rows.append(dict(정의=name, median_setup_sigma_px=float(setup_sigma_px.median()),
                          median_setup_sigma_um_물리스케일환산=float(setup_sigma_px.median() * PHYSICAL_SCALE * 1000)))
    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "within_mount_repeatability.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(out.round(3).to_string(index=False))
    print("\n이 값은 '같은 장착, 10장 반복'의 흔들림이다(재장착 변동의 대리값이 아니다, 규칙 8 "
          "참고). calibration_loso.csv의 방식 간 MAE 차이(특히 1µm 안쪽인 경우)를 이 값과 "
          "나란히 두고 읽어야 한다.")


if __name__ == "__main__":
    main()
