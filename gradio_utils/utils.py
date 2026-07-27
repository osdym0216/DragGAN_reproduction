import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload",
                         tool="sketch",
                         interactive=False,
                         **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"
                                                     ] and type(x) != dict:
            decode_image = gr.processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.ones((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


def get_valid_mask(mask: np.ndarray):
    """Convert mask from gr.Image(0 to 255, RGBA) to binary mask.
    """
    if mask.ndim == 3:
        mask_pil = Image.fromarray(mask).convert('L')
        mask = np.array(mask_pil)
    if mask.max() == 255:
        mask = mask / 255
    return mask


def draw_points_on_image(image,
                         points,
                         curr_point=None,
                         highlight_all=True,
                         radius_scale=0.01):
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():
        if ((curr_point is not None and curr_point == point_key)
                or highlight_all):
            p_color = (255, 0, 0)
            t_color = (0, 0, 255)

        else:
            p_color = (255, 0, 0, 35)
            t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * radius_scale)

        p_start = point.get("start_temp", point["start"])
        p_target = point["target"]

        if p_start is not None and p_target is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            t_draw = int(p_target[0]), int(p_target[1])

            overlay_draw.line(
                (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
                fill=(255, 255, 0),
                width=2,
            )

        if p_start is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            overlay_draw.ellipse(
                (
                    p_draw[0] - rad_draw,
                    p_draw[1] - rad_draw,
                    p_draw[0] + rad_draw,
                    p_draw[1] + rad_draw,
                ),
                fill=p_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(p_draw, "p", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(p_draw, "p", align="center", fill=(0, 0, 0))

        if p_target is not None:
            t_draw = int(p_target[0]), int(p_target[1])
            overlay_draw.ellipse(
                (
                    t_draw[0] - rad_draw,
                    t_draw[1] - rad_draw,
                    t_draw[0] + rad_draw,
                    t_draw[1] + rad_draw,
                ),
                fill=t_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(t_draw, "t", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(t_draw, "t", align="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones(
                (im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"),
                                 im_mask_rgba).convert("RGB")


def on_change_single_global_state(keys,
                                  value,
                                  global_state,
                                  map_transform=None):
    if map_transform is not None:
        value = map_transform(value)

    curr_state = global_state
    if isinstance(keys, str):
        last_key = keys

    else:
        for k in keys[:-1]:
            curr_state = curr_state[k]

        last_key = keys[-1]

    curr_state[last_key] = value
    return global_state


def get_latest_points_pair(points_dict):
    if not points_dict:
        return None
    point_idx = list(points_dict.keys())
    latest_point_idx = max(point_idx)
    return latest_point_idx

def polygon_to_mask(vertices, height, width):
    """クリック頂点リスト [(x,y),...] から 0/1 マスク (H,W)。内部=1, 外部=0。"""
    mask_img = Image.new('L', (width, height), 0)
    if len(vertices) >= 3:
        draw = ImageDraw.Draw(mask_img)
        draw.polygon([(int(x), int(y)) for x, y in vertices], fill=1)
    return np.array(mask_img, dtype=np.uint8)


def draw_polygon_on_image(image, vertices):
    """囲み途中の頂点と辺を可視化して返す。"""
    overlay = Image.new("RGBA", image.size, 0)
    d = ImageDraw.Draw(overlay)
    if len(vertices) >= 2:
        d.line([(int(x), int(y)) for x, y in vertices], fill=(0, 255, 0), width=2)
    for (x, y) in vertices:
        r = 4
        d.ellipse((x-r, y-r, x+r, y+r), fill=(0, 255, 0))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

def auto_mask_from_points(points_dict, height, width, x_factor=2.0):
    """points_dict(global_state['points'])の各 handle/target ペアについて、
    中点中心・(handle-target距離 × x_factor)を直径とする円を可動(1)にする。
    複数ペアがあれば円の和集合。可動=1 の 0/1 マスク (H,W) を返す。"""
    yy, xx = np.mgrid[0:height, 0:width]
    movable = np.zeros((height, width), dtype=np.uint8)
    any_pair = False
    for _, p in points_dict.items():
        start = p.get('start_temp', p.get('start'))
        target = p.get('target')
        if start is None or target is None:
            continue
        any_pair = True
        hx, hy = float(start[0]), float(start[1])
        tx, ty = float(target[0]), float(target[1])
        cx, cy = (hx + tx) / 2.0, (hy + ty) / 2.0
        dist = np.hypot(tx - hx, ty - hy)
        radius = max((dist * x_factor) / 2.0, 5.0)   # 最低半径5px(距離0でも潰れない)
        movable[((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2] = 1
    if not any_pair:
        # ペアが無ければ全体可動(マスク無しと同じ)
        movable[:] = 1
    return movable