"""E10 실행 래퍼 — 10/21 3차 촬영 (PREREG 9절).

원본 E10(pretrain.py / finetune.py / evaluate.py)과 E7·E6 모듈을 수정하지 않고 import한 뒤
아래 값만 바꿔서 원본 main()을 그대로 호출한다.
  - 물리 스케일 0.0367 -> 16 / 핀px 중앙값 (repeatability_ratio.json)
  - V2 계수 10 -> 16 (Ø16 기준물)
  - 사전학습 하단 블러 증강 [0.5, 1.0] -> [] (10/21에서 미검증)
  - 해석 규칙: 기준선 3.4um -> 10/21 동일장착 반복성, fold 승수 7/10 -> 7/9, 엣지 이동 1px -> 1.41px(36.7um 기준)
사용: python3 e10_1021.py {pretrain|finetune|evaluate|time} [원본 인자...]
"""
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(CODE)  # .../20261021_3차_E6_E10
REPO2ND = os.environ.get('REPO2ND', os.path.normpath(os.path.join(
    BASE, '..', '..', '2nd')))  # experiments/2nd
E10 = os.path.join(REPO2ND, 'E10_diam_finetune')
sys.path.insert(0, E10)

import data as D  # noqa: E402  (sys.path에 E7/E6 추가)
import augment as A  # noqa: E402
import calibration as C  # noqa: E402

RR = json.load(open(os.path.join(BASE, 'results', 'repeatability_ratio.json')))
SCALE = RR['V1_scale_mm_per_px']
PIN_D = 16.0
WITHIN_MOUNT_UM = RR['within_mount']['cross50']['median_sigma_um']
EDGE_SHIFT_LIMIT_PX = 0.0367 / SCALE   # 10/08의 1px(36.7um)을 um 기준으로 유지
N_FOLD = 9
WIN_NEED = 7

C.PHYSICAL_SCALE = SCALE
D.PHYSICAL_SCALE_MM_PER_PX = SCALE
A.BOTTOM_ALLOWED['blur_sigma'] = []

OUT = os.environ.get('E10_OUT', 'E10')  # 결과 하위 폴더 이름(스모크 테스트는 다른 이름)
IO = ['--profiles', os.path.join(BASE, 'cache', 'profiles_1021.npz'),
      '--meta', os.path.join(BASE, 'results', 'E10_meta_1021.csv'),
      '--pin-col', 'pin_cross50_px', '--label-source', 'micrometer',
      '--cache-dir', os.path.join(BASE, 'cache', OUT),
      '--results-dir', os.path.join(BASE, 'results', OUT)]


def make_meta():
    import pandas as pd
    v = pd.read_csv(os.path.join(BASE, 'results', 'values_by_image_1021.csv'), encoding='utf-8-sig')
    v = v[v.specimen != 'NO1']
    m = v[['filename', 'bundle', 'specimen', 'role', '실측mm', 'pin_cross50_px', 'mount']].copy()
    m.to_csv(os.path.join(BASE, 'results', 'E10_meta_1021.csv'), index=False, encoding='utf-8-sig')


def patch_finetune(F):
    import numpy as np

    def convert_mm(D_img, pin_img, v_scale, c):
        if v_scale == 'V1':
            return D.PHYSICAL_SCALE_MM_PER_PX * D_img + c
        if v_scale == 'V2':
            return PIN_D * D_img / pin_img + c
        raise ValueError(v_scale)

    def init_c(D_img_np, mm_np, pin_np, v_scale):
        pred0 = D.PHYSICAL_SCALE_MM_PER_PX * D_img_np if v_scale == 'V1' else PIN_D * D_img_np / pin_np
        return float(np.mean(mm_np - pred0))

    F.convert_mm = convert_mm
    F.init_c = init_c


def patch_evaluate(E):
    E.WITHIN_MOUNT_REPEATABILITY_UM = WITHIN_MOUNT_UM

    def apply_interpretation(summary_row, edge_shift_row):
        cond1 = summary_row['평균찍기대비_um'] <= -WITHIN_MOUNT_UM
        cond2 = summary_row['fold_win_count'] >= WIN_NEED
        cond3 = (edge_shift_row['top_shift_median_px'] <= EDGE_SHIFT_LIMIT_PX) and \
                (edge_shift_row['bot_shift_median_px'] <= EDGE_SHIFT_LIMIT_PX)
        verdict = ('실측 레이블이 정보를 담고 있을 가능성. 다음 촬영에서 재확인'
                   if (cond1 and cond2 and cond3) else '이 데이터에서는 실측 레이블 학습이 셋업 잡음을 넘지 못함')
        return dict(cond1_mae_beats_repeatability=cond1, cond2_win_7of9=cond2,
                    cond3_edge_shift_le_1p41px=cond3, verdict=verdict)

    E.apply_interpretation = apply_interpretation


if __name__ == '__main__':
    cmd, rest = sys.argv[1], sys.argv[2:]
    make_meta()
    if cmd == 'time':
        import time
        import numpy as np
        import train_eval as T
        cache, row_df = D.load_merged_cache(IO[1], IO[3], 'pin_cross50_px', 'micrometer')
        m = (row_df.role != 'ng').to_numpy()
        t = time.time()
        T.train_net(cache['top_profiles'][m], cache['top_label'][m], edge='top', head='heatmap', epochs=1)
        print('1 epoch top', round(time.time() - t, 1), 's, n=', int(m.sum()), 'device', T.DEVICE)
    elif cmd == 'pretrain':
        import pretrain as P
        sys.argv = ['pretrain.py'] + IO + rest
        P.main()
    elif cmd == 'finetune':
        import finetune as F
        patch_finetune(F)
        sys.argv = ['finetune.py'] + IO + rest
        F.main()
    elif cmd == 'evaluate':
        import evaluate as E
        patch_evaluate(E)
        sys.argv = ['evaluate.py', '--results-dir', os.path.join(BASE, 'results', OUT)] + rest
        E.main()
    print('limits: within_mount_um', round(WITHIN_MOUNT_UM, 3), 'edge_shift_px', round(EDGE_SHIFT_LIMIT_PX, 3), 'scale', SCALE)
