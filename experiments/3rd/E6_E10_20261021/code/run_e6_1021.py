"""E6 캘리브레이션 — 10/21 3차 촬영 (PREREG 8절).

원본 E6(calibration.py)의 fit_const / fit_linear / fit_fixed_scale / judge / direction 을 import해서
그대로 쓰고, 10/08 전용 부분(묶음 번호, NO1 중복 처리, all10/last5/no7excl, 팀원A 분할)만 10/21 구조로 바꿨다.

- 단위: 시편. 정상 9개(NO2~NO10, NO9는 경계군), NG 2개(NO11, NO12). NO1은 민감도 확인용 예측만.
- 시편 대표값: 그 시편 이미지들의 중앙값(원 코드 규칙). 선택: all(본시험+재장착, 주), burst(본시험만, 보조).
- 방식: 0 평균찍기, 2 전체선형, 3 물리스케일고정(V1, 16/핀px 중앙값), 4 핀비율정규화선형(원본 방식4).
  5 핀비율고정(16 x 베어링/핀 + c)는 E10 V2와 짝을 맞춘 **사전등록 외 보조** 행이다.
- 분할 A: 시편 LOSO(주). 분할 B: CAL(NO2·NO3) 학습 -> NO4~NO10 평가(참고, 2점).
- 1장 판정: 각 fold의 식을 held-out 시편의 이미지 한 장씩에 적용.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO2ND = os.environ.get('REPO2ND', os.path.normpath(os.path.join(_BASE, '..', '..', '2nd')))
sys.path.insert(0, os.path.join(REPO2ND, 'E6_calibration'))
import calibration as C  # noqa: E402  (원본, 수정 안 함)

HERE = _BASE  # 실험 폴더(code/의 상위). results/를 여기서 읽고 쓴다
RR = json.load(open(os.path.join(HERE, 'results', 'repeatability_ratio.json')))
C.PHYSICAL_SCALE = RR['V1_scale_mm_per_px']          # 0.0367 -> 10/21 핀 기준 스케일
PIN_D = 16.0
NORMAL = ['NO%d' % k for k in range(2, 11)]
NG = ['NO11', 'NO12']
BOUNDARY = {'NO9'}
CAL = ['NO2', 'NO3']
WITHIN_MOUNT_UM = RR['within_mount']['cross50']['median_sigma_um']
DEFS = {'cross50': 'bearing_cross50_px', 'grad_peak': 'bearing_grad_peak_px', 'moment': 'bearing_moment_px'}
PIN_COL = {'cross50': 'pin_cross50_px', 'grad_peak': 'pin_grad_peak_px', 'moment': 'pin_moment_px'}


def fit_ratio_fixed(train):
    """방식5(보조): mm = 16 x ratio + c (E10 V2와 같은 꼴)."""
    b = float((train['실측mm'] - PIN_D * train['px']).mean())
    return lambda r: float(PIN_D * r + b), dict(slope=PIN_D, intercept=b)


METHODS = [('0_평균찍기', C.fit_const, 'px'), ('2_전체선형회귀', C.fit_linear, 'px'),
           ('3_물리스케일고정(V1)', C.fit_fixed_scale, 'px'), ('4_핀비율정규화선형', C.fit_linear, 'ratio'),
           ('5_핀비율고정16(보조)', fit_ratio_fixed, 'ratio')]


def tables(img, bcol, pcol):
    img = img.copy()
    img['px'] = img[bcol]
    img['ratio'] = img[bcol] / img[pcol]
    agg = img.groupby('specimen').agg(실측mm=('실측mm', 'first'), px=('px', 'median'), ratio=('ratio', 'median'),
                                      n=('px', 'count'))
    return img, agg


def table_for(agg, kind):
    t = agg[['실측mm', kind]].rename(columns={kind: 'px'})
    return t


def eval_units(fn, agg_t, units):
    rows = []
    for u in units:
        pred = fn(float(agg_t.loc[u, 'px']))
        true = float(agg_t.loc[u, '실측mm'])
        rows.append(dict(unit=u, 실측mm=true, pred_mm=pred, err_um=(pred - true) * 1000, abs_err_um=abs(pred - true) * 1000,
                         판정=C.judge(pred), 방향=C.direction(pred)))
    return rows


def eval_images(fn, img, units, kind):
    rows = []
    for _, r in img[img.specimen.isin(units)].iterrows():
        pred = fn(float(r[kind]))
        rows.append(dict(unit=r.specimen, filename=r.filename, mount=int(r['mount']), 실측mm=r['실측mm'], pred_mm=pred,
                         err_um=(pred - r['실측mm']) * 1000, 판정=C.judge(pred), 방향=C.direction(pred)))
    return rows


def ng_level(pred, true):
    """PREREG 2절 3단계: 방향 = 공칭 20.020에서 벗어난 부호."""
    want = np.sign(true - 20.020)
    got = np.sign(pred - 20.020)
    if C.judge(pred) == 'NG' and got == want:
        return '①NG판정+방향일치'
    if got == want:
        return '②방향만일치(OK판정)'
    return '③방향틀림'


def main():
    df = pd.read_csv(os.path.join(HERE, 'results', 'values_by_image_1021.csv'), encoding='utf-8-sig')
    summary, detail, per_image = [], [], []
    no1_rows = []
    for dname, bcol in DEFS.items():
        pcol = PIN_COL[dname]
        for sel in ('all', 'burst'):
            d = df if sel == 'all' else df[df['mount'] == 0]
            img, agg = tables(d[d.specimen != 'NO1'], bcol, pcol)
            img1, agg1 = tables(d[d.specimen == 'NO1'], bcol, pcol)
            # fold별 평균찍기 오차 (규칙 ② 비교용)
            base_fold = {}
            for held in NORMAL:
                tr = table_for(agg, 'px').loc[[u for u in NORMAL if u != held]]
                fn, _ = C.fit_const(tr)
                base_fold[held] = eval_units(fn, table_for(agg, 'px'), [held])[0]['abs_err_um']
            base_mae = float(np.mean(list(base_fold.values())))

            for mname, fitter, kind in METHODS:
                t = table_for(agg, kind)
                # --- 분할 A: LOSO ---
                rows, img_rows = [], []
                for held in NORMAL:
                    fn, _ = fitter(t.loc[[u for u in NORMAL if u != held]])
                    rows += eval_units(fn, t, [held])
                    img_rows += eval_images(fn, img, [held], kind)
                fn_all, params = fitter(t.loc[NORMAL])
                ng_rows = eval_units(fn_all, t, NG)
                img_rows += eval_images(fn_all, img, NG, kind)
                # NO1 민감도
                if len(agg1):
                    t1 = table_for(agg1, kind)
                    r1 = eval_units(fn_all, t1, ['NO1'])[0]
                    no1_rows.append(dict(정의=dname, 사진선택=sel, 방식=mname, **r1))
                normal_eval = [r for r in rows]
                mae = float(np.mean([r['abs_err_um'] for r in normal_eval]))
                wins = sum(1 for r in normal_eval if r['abs_err_um'] < base_fold[r['unit']])
                mis = sum(1 for r in normal_eval if r['unit'] not in BOUNDARY and r['판정'] == 'NG')
                img_norm = [r for r in img_rows if r['unit'] in NORMAL and r['unit'] not in BOUNDARY]
                img_ng = [r for r in img_rows if r['unit'] in NG]
                s = dict(정의=dname, 사진선택=sel, 분할='A_시편LOSO', 방식=mname, n_normal=len(normal_eval),
                         MAE_um=mae, MAX_um=float(np.max([r['abs_err_um'] for r in normal_eval])),
                         평균찍기대비_um=mae - base_mae, fold_wins=wins,
                         규칙1_통과=bool(mae - base_mae <= -WITHIN_MOUNT_UM), 규칙2_통과=bool(wins >= 7),
                         정상오판정수_경계군제외=mis,
                         이미지_정상오판정률=float(np.mean([r['판정'] == 'NG' for r in img_norm])),
                         slope=params.get('slope'), intercept=params.get('intercept'))
                for r in ng_rows:
                    tag = 'NG1' if r['unit'] == 'NO11' else 'NG2'
                    s[f'{tag}_pred_mm'] = round(r['pred_mm'], 4)
                    s[f'{tag}_판정'] = r['판정']
                    s[f'{tag}_방향'] = r['방향']
                    s[f'{tag}_단계'] = ng_level(r['pred_mm'], r['실측mm'])
                    s[f'{tag}_이미지검출률'] = float(np.mean([x['판정'] == 'NG' for x in img_ng if x['unit'] == r['unit']]))
                summary.append(s)
                detail += [dict(**r, 정의=dname, 사진선택=sel, 분할='A_시편LOSO', 방식=mname) for r in rows + ng_rows]
                per_image += [dict(**r, 정의=dname, 사진선택=sel, 분할='A_시편LOSO', 방식=mname) for r in img_rows]

                # --- 분할 B: CAL 2점 -> NO4~NO10 (참고) ---
                fnb, pb = fitter(t.loc[CAL])
                ev = [u for u in NORMAL if u not in CAL]
                rows_b = eval_units(fnb, t, ev)
                ngb = eval_units(fnb, t, NG)
                fnc, _ = C.fit_const(table_for(agg, 'px').loc[CAL])
                base_b = float(np.mean([r['abs_err_um'] for r in eval_units(fnc, table_for(agg, 'px'), ev)]))
                sb = dict(정의=dname, 사진선택=sel, 분할='B_CAL2점(참고)', 방식=mname, n_normal=len(rows_b),
                          MAE_um=float(np.mean([r['abs_err_um'] for r in rows_b])),
                          MAX_um=float(np.max([r['abs_err_um'] for r in rows_b])),
                          평균찍기대비_um=float(np.mean([r['abs_err_um'] for r in rows_b])) - base_b,
                          정상오판정수_경계군제외=sum(1 for r in rows_b if r['unit'] not in BOUNDARY and r['판정'] == 'NG'),
                          slope=pb.get('slope'), intercept=pb.get('intercept'))
                for r in ngb:
                    tag = 'NG1' if r['unit'] == 'NO11' else 'NG2'
                    sb[f'{tag}_pred_mm'] = round(r['pred_mm'], 4)
                    sb[f'{tag}_판정'] = r['판정']
                    sb[f'{tag}_방향'] = r['방향']
                    sb[f'{tag}_단계'] = ng_level(r['pred_mm'], r['실측mm'])
                summary.append(sb)
                detail += [dict(**r, 정의=dname, 사진선택=sel, 분할='B_CAL2점(참고)', 방식=mname) for r in rows_b + ngb]

    os.makedirs(os.path.join(HERE, 'results', 'E6'), exist_ok=True)
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(HERE, 'results', 'E6', 'calibration_loso_1021.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(detail).to_csv(os.path.join(HERE, 'results', 'E6', 'calibration_by_specimen_1021.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(per_image).to_csv(os.path.join(HERE, 'results', 'E6', 'per_image_judgement_1021.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(no1_rows).to_csv(os.path.join(HERE, 'results', 'E6', 'no1_sensitivity_1021.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['정의', '방식', 'MAE_um', 'MAX_um', '평균찍기대비_um', 'fold_wins', '규칙1_통과', '규칙2_통과', '정상오판정수_경계군제외',
            'NG1_pred_mm', 'NG1_단계', 'NG2_pred_mm', 'NG2_단계']
    a = S[(S['분할'] == 'A_시편LOSO') & (S['사진선택'] == 'all')]
    print(f'V1 scale {C.PHYSICAL_SCALE:.6f} mm/px, 동일장착 기준선 {WITHIN_MOUNT_UM:.2f}um')
    print(a[cols].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
