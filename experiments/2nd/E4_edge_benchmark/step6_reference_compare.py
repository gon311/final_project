"""E4 6~7단계(지시서 (참조) 항목): 팀원A V3 엔진 + 팀원B M0~M5를 참조 열로 붙인다.
둘 다 teacher 후보가 아니라 "우리 정의들이 다른 접근과 같은 자릿수인가" 대조용이다.

  6. **팀원A V3 엔진**: `HEEYEON_ENGINE_V3`(저장소 밖, 수정 안 함)를 같은 ROI
     (`paths.REFERENCE_BOX`)로 직접 import해서 돌린다(YOLO 없이 `--box` 수동 지정과
     같은 경로). 매 이미지가 전체 해상도 오버레이 jpg를 하나씩 만들어서(엔진 자체 동작,
     수정 안 함) 130장이면 ~650MB가 나온다 — `cache/e4/teamA_v3_out/`에 쓰고 스칼라만
     뽑은 뒤 그 폴더는 지운다(0.6절: 대용량 중간 산출물 커밋 금지, 디스크도 정리).
  2. **팀원B M0~M5**: `MINWOOK_ROOT/1008_methods/_out/methods_per_image.csv`를 파일명으로
     붙인다 — **다시 계산하지 않는다**(0.4절).

산출물: results/reference_compare.csv (파일별 cross50/grad_peak/moment/esf1 + teamA_v3 +
teamB M0~M5, 묶음 단위로 묶어 중앙값도 같이 적는다).
"""
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shaftlab import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEAMA_OUT = os.path.join(HERE, "..", "cache", "e4", "teamA_v3_out")


def load_teamA_engine():
    spec = importlib.util.spec_from_file_location("teamA_v3_engine", paths.HEEYEON_ENGINE_V3)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teamA_v3_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_teamA(files):
    mod = load_teamA_engine()
    out_dir = Path(TEAMA_OUT)
    rows = []
    for k, fn in enumerate(files):
        t0 = time.time()
        res = mod.measure(Path(os.path.join(paths.IMAGE_DIR, fn)), paths.REFERENCE_BOX, float("nan"),
                           {"a": 0.0, "b": 0.0}, out_dir)
        rows.append(dict(file=fn, teamA_v3_measured_px=res["measured_pixel"],
                          teamA_v3_std_px=res["std_pixel"], teamA_v3_valid_n=res["valid_point_count"]))
        if k % 20 == 0:
            print(k, fn, round(time.time() - t0, 2), flush=True)
    shutil.rmtree(out_dir, ignore_errors=True)  # 오버레이jpg 등 ~650MB, 스칼라만 남기고 지운다
    return pd.DataFrame(rows)


def load_teamB():
    d = pd.read_csv(os.path.join(paths.MINWOOK_ROOT, "1008_methods", "_out", "methods_per_image.csv"),
                     encoding="utf-8-sig")
    d = d.rename(columns={"파일": "file"})
    wide = d.pivot_table(index="file", columns="방법", values="지름px", aggfunc="first")
    wide.columns = [f"teamB_{c}" for c in wide.columns]
    return wide.reset_index()


def main():
    cols = pd.read_csv(os.path.join(HERE, "..", "cache", "e4", "bearing_columns.csv"), encoding="utf-8-sig")
    our_defs = {"cross50": ("top_cross", "bot_cross"), "grad_peak": ("top_peak", "bot_peak"),
                "moment": ("top_moment", "bot_moment"), "esf1": ("top_esf1", "bot_esf1")}
    per_img = cols.groupby("file").agg(묶음=("묶음", "first"), 시편=("시편", "first"),
                                        실측mm=("실측mm", "first"), 초점=("초점", "first"))
    for name, (t, b) in our_defs.items():
        cols["d_tmp"] = cols[b] - cols[t]
        per_img[f"our_{name}_px"] = cols.groupby("file")["d_tmp"].median()
    per_img = per_img.reset_index()

    files = sorted(per_img["file"])
    teamA = run_teamA(files)
    teamB = load_teamB()

    merged = per_img.merge(teamA, on="file", how="left").merge(teamB, on="file", how="left")
    out_path = os.path.join(HERE, "results", "reference_compare.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(merged)}행)\n")

    setup = merged.groupby("묶음").median(numeric_only=True)
    pd.set_option("display.width", 240)
    diff_cols = [c for c in setup.columns if c.startswith("our_") or c.startswith("teamA_") or c.startswith("teamB_")]
    print("셋업별 중앙값(px), 우리 정의 vs 참조:")
    print(setup[diff_cols].round(2).to_string())

    diam_cols = [c for c in diff_cols if c not in ("our_cross50_px", "teamA_v3_std_px", "teamA_v3_valid_n")]
    print("\n체계적 차이(전체 이미지, our_cross50 대비 median):")
    for c in diam_cols:
        diff = (merged[c] - merged["our_cross50_px"]).median()
        print(f"  {c}: {diff:+.3f}px")


if __name__ == "__main__":
    main()
