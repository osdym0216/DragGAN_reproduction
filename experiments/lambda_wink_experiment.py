"""
experiments/lambda_wink_experiment.py  (mediapipe 版)

片目だけを閉じる編集における「マスク正則化 lambda のトレードオフ破綻」を定量化する。
- 対象の目(閉じたい方)の EAR がどれだけ下がるか
- 非対象の目(マスクで保持したい方)の EAR がどれだけ保たれるか
を lambda ごとに測り、「両立するちょうどいい lambda が存在しないこと」を示す。

ランドマーク検出は mediapipe FaceMesh(ビルド不要・高速)を使う。

前提:
- viz.renderer.Renderer を直接呼ぶ(GUI 非依存)
- bias_act 等の融合カーネルはこのカーネルで有効化済みであること
- pip install mediapipe 済み

使い方(リポジトリ直下で):
    import experiments.lambda_wink_experiment as exp
    recs = exp.run_experiment()
    exp.summarize_and_plot(recs)
"""

import os
import glob
import numpy as np
import torch
from PIL import Image

import mediapipe as mp

import dnnlib
from viz.renderer import Renderer

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
DEVICE = 'cuda'
CACHE_DIR = './checkpoints'
PKL_KEY = 'stylegan2-ffhq-512x512'      # 顔モデル(FFHQ)。ランドマーク検出が効くため必須
OUT_DIR = './experiments/out_lambda_wink'

# mediapipe FaceMesh(468 点)の目のランドマーク。
# EAR 計算に使う 6 点を dlib と同じ意味順 [左端, 上1, 上2, 右端, 下2, 下1] で並べる。
# mediapipe 標準の目のキーポイント(水平端と上下まぶた)を採用。
# 右目(画像の左側に写る、被写体の右目)
RIGHT_EYE_6 = [33, 160, 158, 133, 153, 144]
# 左目(画像の右側に写る、被写体の左目)
LEFT_EYE_6 = [362, 385, 387, 263, 373, 380]

_face_mesh = None


def _init_mp():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3)


def get_eye_points(img_rgb_uint8):
    """RGB uint8 (H,W,3) から、両目の EAR 用 6 点を画素座標 (x,y) で返す。
    戻り値: dict(left=(6,2), right=(6,2)) または None(顔未検出)。"""
    _init_mp()
    H, W = img_rgb_uint8.shape[:2]
    res = _face_mesh.process(img_rgb_uint8)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark  # 正規化座標(0..1)

    def pick(idx_list):
        pts = np.array([[lm[i].x * W, lm[i].y * H] for i in idx_list],
                       dtype=np.float32)
        return pts

    return dict(left=pick(LEFT_EYE_6), right=pick(RIGHT_EYE_6))


def eye_aspect_ratio(eye6):
    """EAR = (縦2本の平均) / 横。並びは [端0, 上1, 上2, 端3, 下4, 下5]。小さいほど閉じ。"""
    p = eye6
    vert = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
    horiz = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(vert / horiz)


# --------------------------------------------------------------------------
# DragGAN ハーネス(GUI 非依存)
# --------------------------------------------------------------------------
def _ckpt_path(key):
    cands = glob.glob(os.path.join(CACHE_DIR, key + '*'))
    assert len(cands) >= 1, f'checkpoint not found for key: {key}'
    return cands[0]


def setup(seed, trunc_psi=0.7, lr=0.001):
    r = Renderer(disable_timing=True)
    res = dnnlib.EasyDict()
    r.init_network(res, pkl=_ckpt_path(PKL_KEY), w0_seed=seed, w_load=None,
                   w_plus=True, noise_mode='const', trunc_psi=trunc_psi,
                   trunc_cutoff=None, input_transform=None, lr=lr)
    r._render_drag_impl(res, is_drag=False, to_pil=False)
    init_img = res.image.detach().cpu().numpy()  # (H,W,3) uint8
    return r, res, init_img


def run_drag(r, res, handle_yx, target_yx, mask, motion_lambda,
             max_steps=300, r1=3, r2=12, trunc_psi=0.7, feature_idx=5):
    """1 回の drag を回して最終画像(RGB uint8)と実行ステップ数を返す。
    mask: (H,W) tensor。_render_drag_impl の mask 引数(固定=1 側)にそのまま渡す。"""
    points = [list(map(float, p)) for p in handle_yx]
    targets = [list(map(float, t)) for t in target_yx]
    steps = max_steps
    for step in range(max_steps):
        r._render_drag_impl(res, points=points, targets=targets, mask=mask,
                            lambda_mask=motion_lambda, reg=0,
                            feature_idx=feature_idx, r1=r1, r2=r2,
                            trunc_psi=trunc_psi, is_drag=True, to_pil=False)
        if res.get('stop', False):
            steps = step + 1
            break
    final_img = res.image.detach().cpu().numpy()
    return final_img, steps


# --------------------------------------------------------------------------
# 実験本体
# --------------------------------------------------------------------------
def build_wink_setup(seed):
    """初期画像のランドマークから、片目を閉じる handle/target と保持マスクを自動生成。
    - 対象(閉じたい) = 被写体の左目(画像右側)
    - handle: 対象の上まぶた、target: 下まぶた方向(=閉じる)
    - mask(可動=1): 全体可動から、非対象の目(被写体の右目)の周囲だけ固定(0)にする
    戻り値: handle_yx, target_yx, mask(0/1,可動=1), eyes0, init_img
    """
    r, res, init_img = setup(seed)
    eyes0 = get_eye_points(init_img)
    if eyes0 is None:
        raise RuntimeError(f'no face detected at seed={seed}; try another seed')

    H, W = init_img.shape[:2]
    le = eyes0['left']   # 対象(閉じたい)
    re = eyes0['right']  # 非対象(保持)

    upper = (le[1] + le[2]) / 2.0   # 上まぶた (x,y)
    lower = (le[4] + le[5]) / 2.0   # 下まぶた (x,y)
    # ハーネスは (y,x) 順
    handle_yx = [[float(upper[1]), float(upper[0])]]
    target_yx = [[float(lower[1]), float(lower[0])]]

    # 可動=1 のマスク。非対象の目(right)の周囲矩形を 0(固定)にする。
    mask = np.ones((H, W), dtype=np.uint8)
    x0, y0 = re[:, 0].min(), re[:, 1].min()
    x1, y1 = re[:, 0].max(), re[:, 1].max()
    pad = int(0.8 * (x1 - x0) + 1)
    xa, ya = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    xb, yb = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    mask[ya:yb, xa:xb] = 0

    return handle_yx, target_yx, mask, eyes0, init_img


def run_experiment(seeds=(0, 1, 2, 3, 4),
                   lambdas=(5, 10, 20, 40, 60, 80, 100),
                   max_steps=300,
                   invert_mask=True):
    """invert_mask=True のとき _render_drag_impl に (1-mask) を渡す(GUI と同じ挙動)。
    最初の目視確認で向きが逆なら invert_mask=False にする。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []

    for seed in seeds:
        try:
            handle_yx, target_yx, mask, eyes0, init_img = build_wink_setup(seed)
        except RuntimeError as e:
            print(f'skip seed={seed}: {e}')
            continue

        ear_L0 = eye_aspect_ratio(eyes0['left'])   # 対象(閉じたい)
        ear_R0 = eye_aspect_ratio(eyes0['right'])  # 非対象(保持)

        for lam in lambdas:
            r, res, _ = setup(seed)   # feat_refs 等をリセットするため毎回
            m = (1 - mask) if invert_mask else mask
            drag_mask = torch.tensor(m).float()
            final_img, steps = run_drag(r, res, handle_yx, target_yx,
                                        mask=drag_mask, motion_lambda=lam,
                                        max_steps=max_steps)
            eyes1 = get_eye_points(final_img)
            if eyes1 is None:
                print(f'seed={seed} lam={lam}: face lost after edit (broken)')
                ear_L1 = ear_R1 = np.nan
            else:
                ear_L1 = eye_aspect_ratio(eyes1['left'])
                ear_R1 = eye_aspect_ratio(eyes1['right'])

            row = dict(
                seed=seed, lam=lam, steps=steps,
                ear_L0=ear_L0, ear_L1=ear_L1,
                ear_R0=ear_R0, ear_R1=ear_R1,
                target_close=ear_L0 - ear_L1,        # 大きいほど良い
                nontarget_leak=abs(ear_R0 - ear_R1)  # 小さいほど良い
                if not np.isnan(ear_R1) else np.nan,
            )
            records.append(row)
            tc = row['target_close']; nl = row['nontarget_leak']
            print(f"seed={seed} lam={lam:3d} steps={steps:3d} "
                  f"target_close={tc:+.3f} "
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
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, 'lambda_tradeoff.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('saved:', out)
    return lambdas, close_mean, leak_mean


if __name__ == '__main__':
    recs = run_experiment()
    summarize_and_plot(recs)