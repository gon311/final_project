"""E4 3단계(지시서 지표: "같은 정의로 잰 Ø10 핀 px의 F3 내 범위"). E5에서 정한
`pin_relative` 창(핀 왼쪽 끝 xL 기준 s=60~105)으로 잰 핀 지름을, cross50·grad_peak·
moment·esf1 네 정의로 각각 구해 F3(묶음 7~13, 핀이 고정된 뒤 초점만 바뀐 구간) 안에서
얼마나 흔들리는지 본다. 핀은 물리적으로 그 구간 안에서 안 바뀌므로, 범위가 크다는 건
그 정의가 초점 변화(F2->F3)에 더 민감하다는 뜻이다.

**ESF1 적합이 실패한(`*_esf1_ok=False`) 열은 명시적으로 뺀다**(Codex `/codex:review`
반영) — 전에는 실패한 열의 `diam_tmp`가 NaN이 돼 pandas가 `median`/`max`/`min`/`std`에서
조용히 빼면서도 `f3_n`에는 "F3 구간 이미지 수"(전체)를 그대로 적어, 실제로 몇 장이 유효한지
드러나지 않았다. 이제 열 단위로 명시적으로 거르고, 이미지별로 유효 열이 하나도 없으면 그
이미지 자체를 빼며, `f3_n_valid_images`로 실제 쓰인 이미지 수를 따로 적는다.

산출물: results/pin_f3_range.csv
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_DEFS = {
    "cross50": ("top_cross", "bot_cross", None),
    "grad_peak": ("top_peak", "bot_peak", None),
    "moment": ("top_moment", "bot_moment", None),
    "esf1": ("top_esf1", "bot_esf1", ("top_esf1_ok", "bot_esf1_ok")),
}
F3_BUNDLES = {7, 8, 9, 10, 11, 12, 13}


def per_image_diam(cols: pd.DataFrame, tcol: str, bcol: str, ok_cols) -> pd.DataFrame:
    rows = []
    for fn, g in cols.groupby("file"):
        if ok_cols is not None:
            ok = g[ok_cols[0]].to_numpy(bool) & g[ok_cols[1]].to_numpy(bool)
        else:
            ok = pd.Series(True, index=g.index).to_numpy()
        gg = g[ok]
        row0 = g.iloc[0]
        if len(gg) == 0:
            rows.append(dict(file=fn, 묶음=row0["묶음"], 초점=row0["초점"], diam=float("nan"), n_ok=0))
            continue
        rows.append(dict(file=fn, 묶음=row0["묶음"], 초점=row0["초점"],
                          diam=float((gg[bcol] - gg[tcol]).median()), n_ok=len(gg)))
    return pd.DataFrame(rows)


def main():
    cols = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "pin_columns.csv"), encoding="utf-8-sig")

    rows = []
    for name, (tcol, bcol, ok_cols) in PIN_DEFS.items():
        per_img = per_image_diam(cols, tcol, bcol, ok_cols)
        f3_all = per_img[per_img["묶음"].isin(F3_BUNDLES)]
        f3 = f3_all.dropna(subset=["diam"])  # 유효 열이 하나도 없던 이미지는 뺀다
        all13 = per_img.dropna(subset=["diam"])
        rows.append(dict(
            정의=name,
            f3_n_total_images=int(len(f3_all)), f3_n_valid_images=int(len(f3)),
            f3_range_px=float(f3["diam"].max() - f3["diam"].min()) if len(f3) else float("nan"),
            f3_std_px=float(f3["diam"].std()) if len(f3) else float("nan"),
            all_range_px=float(all13["diam"].max() - all13["diam"].min()) if len(all13) else float("nan"),
            f3_median_px=float(f3["diam"].median()) if len(f3) else float("nan"),
        ))
    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results", "pin_f3_range.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}\n")
    print(out.round(4).to_string(index=False))
    print("\nF3(묶음7~13) 안에서 핀 px 범위가 가장 작은 정의일수록 초점 변화에 덜 흔들린다고 "
          "본다(핀은 F3 구간 안에서 물리적으로 고정돼 있어야 한다, E5 3단계). "
          "f3_n_valid_images가 f3_n_total_images보다 작으면 그만큼 ESF1 적합이 실패한 "
          "이미지가 빠졌다는 뜻이다.")


if __name__ == "__main__":
    main()
