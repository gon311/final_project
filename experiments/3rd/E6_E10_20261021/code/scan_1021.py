"""150장 측정(E6 scan_values 대응) + E7 프로파일 추출. 이미지별 결과를 cache/scan에 저장(재시작 가능).

사용: SHAFT_1021_DATA=<사진 폴더> python3 code/scan_1021.py [제한 시간(초), 기본 160]
끝나지 않았으면 같은 명령을 다시 실행하면 이어서 한다. 다 끝나면 assemble_1021.py."""
import glob, os, sys, csv, json, time, unicodedata
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab1021 as L
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get('SHAFT_1021_DATA', '/Volumes/Lexar/Final/_private/20261021_촬영데이터')  # 사진 폴더(저장소 밖, 읽기 전용)
N = lambda s: unicodedata.normalize('NFC', s)
disk = {N(os.path.relpath(p, D)): p for p in glob.glob(D + '/*/*/*.jpg')}
meta = list(csv.DictReader(open(f'{W}/results/meta_1021_all.csv', encoding='utf-8-sig')))
out_dir = f'{W}/cache/scan'; os.makedirs(out_dir, exist_ok=True)  # cache/는 .gitignore 대상
t_end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 160)
done = 0
for m in meta:
    key = os.path.join(m['folder'], m['filename'])
    js = f"{out_dir}/{m['filename']}.json"
    if os.path.exists(js): continue
    if time.time() > t_end: break
    g = cv2.imread(disk[N(key)], cv2.IMREAD_GRAYSCALE)
    off = L.NO1_OFFSET_Q if m['specimen'] == 'NO1' else (0, 0)
    try:
        o, (pr, lb, xs, ce) = L.measure_all(g, offset_q=off)
        np.savez(f"{out_dir}/{m['filename']}.npz", profiles=pr.astype(np.float32), labels=lb, xs=xs, centers=ce)
        o['status'] = 'ok'
    except Exception as e:
        o = dict(status=f'fail: {e}')
    json.dump({k: (bool(v) if isinstance(v, (np.bool_,)) else (float(v) if isinstance(v, (np.floating, np.integer)) else v)) for k, v in o.items()}, open(js, 'w'))
    done += 1
print('done this call', done, 'total json', len(glob.glob(out_dir + '/*.json')))
