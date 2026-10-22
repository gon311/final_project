"""cache/scan의 이미지별 결과 -> results/values_by_image_1021.csv + cache/profiles_1021.npz(E7 형식).

- values_by_image_1021.csv: 메타 + E6 측정값(베어링·핀 cross50/grad_peak/moment 등), 150행
- profiles_1021.npz: E7 build_cache.py와 같은 필드·폭(반창 40 + 버퍼 4 = 89px). NO1은 넣지 않는다(E10 학습·평가 제외).
사용: python3 code/assemble_1021.py   (scan_1021.py가 150장을 다 처리한 뒤)
"""
import collections, csv, json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
meta = list(csv.DictReader(open('results/meta_1021_all.csv', encoding='utf-8-sig')))
rows, bad = [], []
P = collections.defaultdict(list)
for m in meta:
    o = json.load(open(f"cache/scan/{m['filename']}.json"))
    if o['status'] != 'ok':
        bad.append((m['filename'], o['status']))
        continue
    r = dict(m)
    r.update({k: v for k, v in o.items() if k != 'status'})
    rows.append(r)
    if m['specimen'] == 'NO1':
        continue
    z = np.load(f"cache/scan/{m['filename']}.npz")
    n = len(z['xs'])
    P['top_profiles'].append(z['profiles'][:, 0, :]); P['bot_profiles'].append(z['profiles'][:, 1, :])
    P['top_label'].append(z['labels'][:, 0]); P['bot_label'].append(z['labels'][:, 1])
    P['top_center'].append(z['centers'][:, 0]); P['bot_center'].append(z['centers'][:, 1]); P['x'].append(z['xs'])
    for k, v in (('filename', m['filename']), ('bundle', int(m['bundle'])), ('specimen', m['specimen']),
                 ('role', m['role']), ('focus', 'F_1021'), ('pin', 'P16')):
        P[k] += [v] * n
print('fail', bad)
with open('results/values_by_image_1021.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
cat = lambda k, t: np.concatenate(P[k]).astype(t)
out = dict(top_profiles=cat('top_profiles', np.float32), bot_profiles=cat('bot_profiles', np.float32),
           top_label=cat('top_label', np.float32), bot_label=cat('bot_label', np.float32),
           top_center=cat('top_center', np.int32), bot_center=cat('bot_center', np.int32), x=cat('x', np.int32),
           filename=np.array(P['filename']), bundle=np.array(P['bundle'], dtype=np.int32),
           specimen=np.array(P['specimen']), role=np.array(P['role']), focus=np.array(P['focus']), pin=np.array(P['pin']),
           half_window_px=np.array([40]), shift_buffer_px=np.array([4]), col_step_px=np.array([2]))
os.makedirs('cache', exist_ok=True)
np.savez('cache/profiles_1021.npz', **out)
print('rows', len(rows), 'profiles', out['top_profiles'].shape)
