"""00_3차촬영_사진목록 엑셀 -> results/meta_1021_all.csv (PREREG 1·2절의 역할 배정).

- 암막없음(3_) 폴더는 주 분석에서 제외해 메타에 넣지 않는다.
- role: NO1 -> excluded_no1, NO11·NO12 -> ng, NO9 -> boundary(경계군), 나머지는 엑셀 구분(CAL/검증/랜덤)
- mount: 본시험 0, 재장착 1~5. setup_id = 시편_M장착, bundle = 100 x 시편번호 + 장착(10/08 묶음 번호와 겹치지 않게)
- 한글 경로는 NFC로 정규화한다(macOS는 NFD로 저장).

사용: SHAFT_1021_DATA=<사진 폴더> python3 code/make_meta_1021.py
"""
import csv, glob, os, unicodedata
import openpyxl

D = os.environ.get('SHAFT_1021_DATA', '/Volumes/Lexar/Final/_private/20261021_촬영데이터')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = lambda s: unicodedata.normalize('NFC', s)
xlsx = glob.glob(os.path.join(D, '00_*.xlsx'))[0]
ws = openpyxl.load_workbook(xlsx, data_only=True)['파일목록']
out = []
for r in ws.iter_rows(min_row=2, values_only=True):
    if not r[0]:
        continue
    fn, folder, tm, stage, sp, cls, mm, seq, remount, res, zoom, sh, iso, ev, note = r
    folder, stage, cls = N(folder), N(stage), N(cls)
    if folder.startswith('3_'):
        continue
    k = int(sp[2:])
    mount = 0 if folder.startswith('1_') else int(remount)
    if sp == 'NO1':
        role = 'excluded_no1'
    elif sp in ('NO11', 'NO12'):
        role = 'ng'
    elif sp == 'NO9':
        role = 'boundary'
    else:
        role = {'CAL': 'cal', '검증': 'val', '랜덤': 'rand'}[cls]
    out.append(dict(filename=fn, folder=folder, 단계=stage, specimen=sp, 구분=cls, 실측mm=float(mm), 시각=tm,
                    mount=mount, setup_id=f'{sp}_M{mount}', bundle=100 * k + mount, role=role,
                    zoom=zoom, shutter=sh, iso=iso))
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
with open(os.path.join(BASE, 'results', 'meta_1021_all.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), 'rows')
