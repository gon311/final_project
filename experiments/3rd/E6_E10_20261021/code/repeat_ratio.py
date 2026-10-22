"""PREREG 4·6·7절: V1 스케일, 반복성 기준선, 비율 실험(주 보정 방법 결정). 마이크로미터 값은 쓰지 않는다."""
import numpy as np, pandas as pd, json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 실험 폴더 기준 상대경로
df = pd.read_csv('results/values_by_image_1021.csv', encoding='utf-8-sig')
main = df[df.specimen != 'NO1']
burst = main[main.mount == 0]
PIN_D = 16.0
scale = PIN_D / burst['pin_cross50_px'].median()           # mm/px (4절)
out = dict(V1_scale_mm_per_px=scale, pin_median_px=burst['pin_cross50_px'].median())

# 6절 동일 장착: 정상 NO2~NO10 본시험 10장 σ 중앙값
normal = ['NO%d' % k for k in range(2, 11)]
wm = {}
for d in ('cross50', 'grad_peak', 'moment'):
    s = burst[burst.specimen.isin(normal)].groupby('specimen')[f'bearing_{d}_px'].std()
    wm[d] = dict(median_sigma_px=float(s.median()), median_sigma_um=float(s.median() * scale * 1000))
out['within_mount'] = wm

# 6절 재장착: NO7~NO12 6장착
rm = main[main.specimen.isin(['NO%d' % k for k in range(7, 13)])]
mv = rm.groupby(['specimen', 'mount']).agg(b=('bearing_cross50_px', 'median'), p=('pin_cross50_px', 'median')).reset_index()
mv['ratio_mm'] = PIN_D * mv.b / mv.p
sd_raw = mv.groupby('specimen').b.std() * scale * 1000
sd_ratio = mv.groupby('specimen').ratio_mm.std() * 1000
sd_pin = mv.groupby('specimen').p.std()
out['remount'] = dict(per_specimen_sd_raw_um=sd_raw.round(2).to_dict(), per_specimen_sd_ratio_um=sd_ratio.round(2).to_dict(),
                      median_sd_raw_um=float(sd_raw.median()), median_sd_ratio_um=float(sd_ratio.median()),
                      median_sd_pin_px=float(sd_pin.median()))

# 7절 비율 실험
mv['db'] = mv.b - mv.groupby('specimen').b.transform('mean')
mv['dp'] = mv.p - mv.groupby('specimen').p.transform('mean')
x, y = mv.dp.to_numpy(), mv.db.to_numpy()
slope = float((x * y).sum() / (x * x).sum())
resid = y - slope * x
dof = len(x) - mv.specimen.nunique() - 1
se = float(np.sqrt((resid ** 2).sum() / dof / (x * x).sum()))
from scipy import stats
tcrit = stats.t.ppf(0.975, dof)
ci = (slope - tcrit * se, slope + tcrit * se)
rng = np.random.default_rng(0); sps = mv.specimen.unique(); bs = []
for _ in range(2000):
    pick = rng.choice(sps, len(sps), replace=True)
    sub = pd.concat([mv[mv.specimen == s] for s in pick])
    xx, yy = sub.dp.to_numpy(), sub.db.to_numpy()
    if (xx * xx).sum() > 0: bs.append((xx * yy).sum() / (xx * xx).sum())
bci = tuple(np.percentile(bs, [2.5, 97.5]))
expected = float(burst[burst.specimen == 'NO2'].bearing_cross50_px.median() / burst[burst.specimen == 'NO2'].pin_cross50_px.median())
half = expected / 2
if ci[0] <= expected <= ci[1] and not (ci[0] <= half <= ci[1]): verdict, primary = 'V2 유효', 'V2'
elif not (ci[0] <= expected <= ci[1]): verdict, primary = '부분/무효 (CI가 기대 기울기 제외)', 'V1'
else: verdict, primary = '판정 불가 (CI가 기대·절반 모두 포함)', 'V1'
out['ratio_experiment'] = dict(n_points=len(x), dof=dof, slope=slope, se=se, ci95_t=ci, ci95_bootstrap=bci,
                               expected=expected, half=half, pin_dp_range_px=(float(x.min()), float(x.max())),
                               verdict=verdict, primary_method=primary)
mv.to_csv('results/ratio_experiment_mounts.csv', index=False, encoding='utf-8-sig')
json.dump(out, open('results/repeatability_ratio.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
