"""
experiments/auto_mask_improvement.py

改善:handle/target 点の中点を中心に、両点間距離 × x_factor を直径とする円だけを
可動領域にする「自動マスク」を導入し、マスク無しで起きる大域崩壊を防げるかを観察する。

- マスク無し / 自動マスク(x_factor 数種)で同じ片目閉じ編集を行い、結果画像を並べて保存
- 崩壊するか否かは目視で判断(頭部テクスチャの崩壊など)
- 補助として、可動域の外側領域が編集前後でどれだけ変化したか(大域変化量)も L1 で出す

実行(リポジトリ直下から):
    python -m experiments.auto_mask_improvement
"""

import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import dnnlib
from viz.renderer import Renderer

CACHE_DIR = './checkpoints'
PKL_KEY = 'stylegan2-ffhq-512x512'
OUT_DIR = './experiments/out_auto_mask'
MODEL_PATH = './experiments/face_landmarker.task'

SEEDS = (0, 1, 2)
X_FACTORS = (None, 1.5, 2.0, 3.0, 4.0)   # None = マスク無し
MAX_STEPS = 300
CLOSE_DISTANCE = 25   # 片目を閉じるドラッグ距離(px)

# 左目(被写体の左目=画像右側)上下まぶた
IDX_LEFT_EYE_UP = 386
IDX_LEFT_EYE_LOW = 374

_landmarker = None


def _init_mp():
    global _landmarker
    if _landmarker is None:
        assert os.path.exists(MODEL_PATH), f'model not found: {MODEL_PATH}'
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base, num_faces=1, min_face_detection_confidence=0.3)
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)


def get_landmarks_xy(img):
    _init_mp()
    H, W = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
    res = _landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)


def _ckpt(key):
    return glob.glob(os.path.join(CACHE_DIR, key + '*'))[0]


def setup(seed, lr=0.001, trunc_psi=0.7):
    r = Renderer(disable_timing=True)
    res = dnnlib.EasyDict()
    r.init_network(res, pkl=_ckpt(PKL_KEY), w0_seed=seed, w_load=None,
                   w_plus=True, noise_mode='const', trunc_psi=trunc_psi,
                   trunc_cutoff=None, input_transform=None, lr=lr)
    r._render_drag_impl(res, is_drag=False, to_pil=False)
    return r, res, res.image.detach().cpu().numpy()


def auto_mask_midpoint(handle_xy, target_xy, H, W, x_factor):
    """中点中心・(距離×x_factor) を直径とする円を可動(1)にする。可動=1 の mask を返す。"""
    hx, hy = handle_xy
    tx, ty = target_xy
    cx, cy = (hx + tx) / 2.0, (hy + ty) / 2.0
    dist = np.hypot(tx - hx, ty - hy)
    radius = (dist * x_factor) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2).astype(np.uint8)


def run_drag(r, res, handle_yx, target_yx, drag_mask, motion_lambda=20,
             max_steps=MAX_STEPS, r1=3, r2=12, trunc_psi=0.7, feature_idx=5):
    points = [[int(round(p[0])), int(round(p[1]))] for p in handle_yx]
    targets = [[int(round(t[0])), int(round(t[1]))] for t in target_yx]
    steps = max_steps
    for step in range(max_steps):
        r._render_drag_impl(res, points=points, targets=targets, mask=drag_mask,
                            lambda_mask=motion_lambda, reg=0, feature_idx=feature_idx,
                            r1=r1, r2=r2, trunc_psi=trunc_psi, is_drag=True, to_pil=False)
        if res.get('stop', False):
            steps = step + 1
            break
    return res.image.detach().cpu().numpy(), steps


def run_experiment(seeds=SEEDS, x_factors=X_FACTORS):
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for seed in seeds:
        _, _, init_img = setup(seed)
        H, W = init_img.shape[:2]
        lm = get_landmarks_xy(init_img)
        if lm is None:
            print(f'skip seed={seed}: no face'); continue
        Image.fromarray(init_img).save(os.path.join(OUT_DIR, f'seed{seed}_init.png'))

        up = lm[IDX_LEFT_EYE_UP]
        low = lm[IDX_LEFT_EYE_LOW]
        direction = np.sign(low[1] - up[1]) or 1.0
        handle_xy = (float(up[0]), float(up[1]))
        target_xy = (float(up[0]), float(up[1] + CLOSE_DISTANCE * direction))
        handle_yx = [[handle_xy[1], handle_xy[0]]]
        target_yx = [[target_xy[1], target_xy[0]]]

        for xf in x_factors:
            r, res, _ = setup(seed)
            if xf is None:
                drag_mask = None                       # マスク無し(全体可動)
                movable = np.ones((H, W), np.uint8)
            else:
                movable = auto_mask_midpoint(handle_xy, target_xy, H, W, xf)
                drag_mask = torch.tensor(1 - movable).float()   # 固定=1 側に反転
            final_img, steps = run_drag(r, res, handle_yx, target_yx, drag_mask)

            # 大域変化量:可動域の外側で、編集前後がどれだけ変わったか(L1, 0..1)
            outside = (movable == 0)
            if outside.sum() > 0:
                diff = np.abs(final_img.astype(np.float32) - init_img.astype(np.float32)) / 255.0
                global_change = float(diff.mean(axis=2)[outside].mean())
            else:
                global_change = np.nan

            tag = 'nomask' if xf is None else f'x{xf}'
            Image.fromarray(final_img).save(
                os.path.join(OUT_DIR, f'seed{seed}_{tag}.png'))
            records.append(dict(seed=seed, x_factor=(0 if xf is None else xf),
                                steps=steps, global_change=round(global_change, 4)))
            print(f"seed={seed} {tag:8s} steps={steps:3d} "
                  f"outside_change(L1)={global_change:.4f}")
    return records


if __name__ == '__main__':
    recs = run_experiment()
    # 簡単なサマリ:x_factor ごとの平均 outside_change
    import collections
    agg = collections.defaultdict(list)
    for r in recs:
        agg[r['x_factor']].append(r['global_change'])
    print('\n=== outside-region change by x_factor (lower = less collapse) ===')
    for xf in sorted(agg):
        label = 'no mask' if xf == 0 else f'x={xf}'
        print(f'  {label:10s} mean outside_change = {np.nanmean(agg[xf]):.4f}')