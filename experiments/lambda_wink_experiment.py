"""
experiments/lambda_wink_experiment.py  (mediapipe tasks / 新 API 統合版)

片目だけを閉じる編集における「マスク正則化 lambda のトレードオフ破綻」を定量化する。
- 対象の目(閉じたい方)の EAR がどれだけ下がるか
- 非対象の目(マスクで保持したい方)の EAR がどれだけ保たれるか
を lambda ごとに測り、「両立するちょうどいい lambda が存在しないこと」を示す。

ランドマーク検出: mediapipe tasks の FaceLandmarker(新 API)。
事前に face_landmarker.task を experiments/ に置くこと:
    wget -O experiments/face_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

実行(リポジトリ直下から 3.11 で):
    python -m experiments.lambda_wink_experiment
"""

import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

# リポジトリ直下を import path に(どこから呼ばれても dnnlib が見えるように)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import dnnlib
from viz.renderer import Renderer

# ==========================================================================
# 設定
# ==========================================================================
DEVICE = 'cuda'
CACHE_DIR = './checkpoints'
PKL_KEY = 'stylegan2-ffhq-512x512'
OUT_DIR = './experiments/out_lambda_wink'
MODEL_PATH = './experiments/face_landmarker.task'

# mediapipe Face Mesh(468 点)の目のランドマーク。
# EAR 用 6 点を [左端, 上1, 上2, 右端, 下2, 下1] の意味順で並べる。
RIGHT_EYE_6 = [33, 160, 158, 133, 153, 144]   # 被写体の右目(画像左側)
LEFT_EYE_6 = [362, 385, 387, 263, 373, 380]   # 被写体の左目(画像右側)

LAMBDAS = (5, 10, 20, 40, 60, 80, 100)
SEEDS = (0, 1, 2, 3, 4)
MAX_STEPS = 300
INVERT_MASK = True   # 最初の目視確認で向きが逆なら False

# ==========================================================================
# ランドマーク(mediapipe tasks 新 API)
# ==========================================================================
_landmarker = None


def _init_mp():
    global _landmarker
    if _landmarker is None:
        assert os.path.exists(MODEL_PATH), (
            f'model not found: {MODEL_PATH}\n'
            '  wget -O experiments/face_landmarker.task '
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task')
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base, num_faces=1,
            min_face_detection_confidence=0.3)
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)


def get_eye_points(img_rgb_uint8):
    """RGB uint8 (H,W,3) から両目の EAR 用 6 点を画素座標で返す。顔未検出なら None。"""
    _init_mp()
    H, W = img_rgb_uint8.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(img_rgb_uint8))
    result = _landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]

    def pick(idx_list):
        return np.array([[lm[i].x * W, lm[i].y * H] for i in idx_list],
                        dtype=np.float32)

    return dict(left=pick(LEFT_EYE_6), right=pick(RIGHT_EYE_6))


def eye_aspect_ratio(eye6):
    """EAR = (縦2本の平均)/横。並び [端0,上1,上2,端3,下4,下5]。小さいほど閉じ。"""
    p = eye6
    vert = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
    horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(vert / horiz)


# ==========================================================================
# DragGAN ハーネス(GUI 非依存)
# ==========================================================================
def _ckpt(key):
    cands = glob.glob(os.path.join(CACHE_DIR, key + '*'))
    assert cands, f'checkpoint not found: {key}'
    return cands[0]


def setup(seed, lr=0.001, trunc_psi=0.7):
    r = Renderer(disable_timing=True)
    res = dnnlib.EasyDict()
    r.init_network(res, pkl=_ckpt(PKL_KEY), w0_seed=seed, w_load=None,
                   w_plus=True, noise_mode='const', trunc_psi=trunc_psi,
                   trunc_cutoff=None, input_transform=None, lr=lr)
    r._render_drag_impl(res, is_drag=False, to_pil=False)
    return r, res, res.image.detach().cpu().numpy()


def run_drag(r, res, handle_yx, target_yx, mask, motion_lambda,
             max_steps=MAX_STEPS, r1=3, r2=12, trunc_psi=0.7, feature_idx=5):
    points = [[int(round(p[0])), int(round(p[1]))] for p in handle_yx]
    targets = [[int(round(t[0])), int(round(t[1]))] for t in target_yx]
    steps = max_steps
    for step in range(max_steps):
        r._render_drag_impl(res, points=points, targets=targets, mask=mask,
                            lambda_mask=motion_lambda, reg=0,
                            feature_idx=feature_idx, r1=r1, r2=r2,
                            trunc_psi=trunc_psi, is_drag=True, to_pil=False)
        if res.get('stop', False):
            steps = step + 1
            break
    return res.image.detach().cpu().numpy(), steps


# ==========================================================================
# 実験本体
# ==========================================================================
def build_wink_setup(seed):
    """初期画像のランドマークから、片目を閉じる handle/target と保持マスクを自動生成。
    対象(閉じたい)=被写体の左目(画像右側)、保持=被写体の右目(画像左側)。
    戻り値: handle_yx, target_yx, mask(可動=1), eyes0, init_img"""
    r, res, init_img = setup(seed)
    eyes0 = get_eye_points(init_img)
    if eyes0 is None:
        raise RuntimeError(f'no face detected at seed={seed}')

    H, W = init_img.shape[:2]
    le, re = eyes0['left'], eyes0['right']

    upper = (le[1] + le[2]) / 2.0
    lower = (le[4] + le[5]) / 2.0
    handle_yx = [[float(upper[1]), float(upper[0])]]   # (y,x)
    target_yx = [[float(lower[1]), float(lower[0])]]

    mask = np.ones((H, W), dtype=np.uint8)
    x0, y0 = re[:, 0].min(), re[:, 1].min()
    x1, y1 = re[:, 0].max(), re[:, 1].max()
    pad = int(0.8 * (x1 - x0) + 1)
    xa, ya = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    xb, yb = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    mask[ya:yb, xa:xb] = 0
    return handle_yx, target_yx, mask, eyes0, init_img


def run_experiment(seeds=SEEDS, lambdas=LAMBDAS, max_steps=MAX_STEPS,
                   invert_mask=INVERT_MASK):
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for seed in seeds:
        try:
            handle_yx, target_yx, mask, eyes0, init_img = build_wink_setup(seed)
        except RuntimeError as e:
            print(f'skip seed={seed}: {e}')
            continue

        ear_L0 = eye_aspect_ratio(eyes0['left'])
        ear_R0 = eye_aspect_ratio(eyes0['right'])
        Image.fromarray(init_img).save(os.path.join(OUT_DIR, f'seed{seed}_init.png'))

        for lam in lambdas:
            r, res, _ = setup(seed)
            m = (1 - mask) if invert_mask else mask
            drag_mask = torch.tensor(m).float()
            final_img, steps = run_drag(r, res, handle_yx, target_yx,
                                        mask=drag_mask, motion_lambda=lam,
                                        max_steps=max_steps)
            eyes1 = get_eye_points(final_img)
            if eyes1 is None:
                print(f'seed={seed} lam={lam}: face lost (broken)')
                ear_L1 = ear_R1 = np.nan
            else:
                ear_L1 = eye_aspect_ratio(eyes1['left'])
                ear_R1 = eye_aspect_ratio(eyes1['right'])

            tc = ear_L0 - ear_L1 if not np.isnan(ear_L1) else np.nan
            nl = abs(ear_R0 - ear_R1) if not np.isnan(ear_R1) else np.nan
            records.append(dict(seed=seed, lam=lam, steps=steps,
                                ear_L0=ear_L0, ear_L1=ear_L1,
                                ear_R0=ear_R0, ear_R1=ear_R1,
                                target_close=tc, nontarget_leak=nl))
            print(f"seed={seed} lam={lam:3d} steps={steps:3d} | "
                  f"target_close={tc if np.isnan(tc) else round(tc,3)}  "
                  f"nontarget_leak={nl if np.isnan(nl) else round(nl,3)}")
            Image.fromarray(final_img).save(
                os.path.join(OUT_DIR, f'seed{seed}_lam{lam}.png'))
    return records


def summarize_and_plot(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lambdas = sorted(set(r['lam'] for r in records))

    def mean_over(key, lam):
        vals = [r[key] for r in records
                if r['lam'] == lam and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return np.mean(vals) if vals else np.nan

    close_mean = [mean_over('target_close', l) for l in lambdas]
    leak_mean = [mean_over('nontarget_leak', l) for l in lambdas]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, close_mean, 'o-', label='target eye closing (want HIGH)')
    ax.plot(lambdas, leak_mean, 's-', label='non-target leak (want LOW)')
    ax.set_xlabel('lambda (mask regularization)')
    ax.set_ylabel('EAR change')
    ax.set_title('One-eye wink: no single lambda satisfies both')
    ax.legend(); ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, 'lambda_tradeoff.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('saved:', out)
    # 数値も CSV で残す
    import csv
    csv_path = os.path.join(OUT_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    print('saved:', csv_path)


if __name__ == '__main__':
    recs = run_experiment()
    if recs:
        summarize_and_plot(recs)
    else:
        print('no records (all seeds failed face detection?)')