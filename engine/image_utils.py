import unicodedata
from typing import Callable, List, Tuple

import cv2
import numpy as np
import einops


def resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
    ratio = float(size) / max(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR_EXACT)


def color_difference(rgb1, rgb2) -> float:
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = (
        cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32)
        - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float32)
    )
    diff[..., 0] *= 0.392
    return np.linalg.norm(diff, axis=2).item()


def is_valuable_char(ch) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return False
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return False
    if ch in (" ", "\t", "\n", "\r") or ord(ch) == 0:
        return False
    if cat == "Zs":
        return False
    if cat in ("Cc", "Cf") and ch not in ("\t", "\n", "\r"):
        return False
    if ch.isdigit():
        return False
    return True


def count_valuable_text(text: str) -> int:
    return sum(1 for ch in text if is_valuable_char(ch))


def is_valuable_text(text: str) -> bool:
    return any(is_valuable_char(ch) for ch in text)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    MULT = 256
    pad_h = 0
    pad_w = 0
    if target_h % MULT != 0:
        pad_h = MULT - target_h % MULT
    if target_w % MULT != 0:
        pad_w = MULT - target_w % MULT
    target_h32 = target_h + pad_h
    target_w32 = target_w + pad_w
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (int(target_w32 / 2), int((target_h + pad_h) / 2)), pad_w, pad_h



def square_pad_resize(img: np.ndarray, tgt_size: int):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
    return img, down_scale_ratio, pad_h, pad_w


def det_rearrange_forward(
    img: np.ndarray,
    dbnet_batch_forward: Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray]],
    tgt_size: int = 1280,
    max_batch_size: int = 4,
    device='cuda',
    verbose=False,
):
    # Rearrange extreme-aspect-ratio images into square batches for the network.
    # Returns (None, None) if rearrangement is not needed.

    def _unrearrange(patch_lst, transpose, channel=1, pad_num=0):
        _psize = _h = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_num
        for ii, p in enumerate(patch_lst):
            if transpose:
                p = einops.rearrange(p, 'c h w -> c w h')
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + _psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t:b, :] += p[..., :b - t, l:r]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t:t + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if transpose:
            tgtmap = einops.rearrange(tgtmap, 'c h w -> c w h')
        return tgtmap[None, ...]

    def _patch2batches(patch_lst, p_num, transpose):
        if transpose:
            patch_lst = einops.rearrange(
                patch_lst, '(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c', p_num=p_num
            )
        else:
            patch_lst = einops.rearrange(
                patch_lst, '(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c', p_num=p_num
            )
        batches = [[]]
        for ii, patch in enumerate(patch_lst):
            if len(batches[-1]) >= max_batch_size:
                batches.append([])
            p, down_scale_ratio, pad_h, pad_w = square_pad_resize(patch, tgt_size=tgt_size)
            assert pad_h == pad_w
            pad_size = pad_h
            batches[-1].append(p)
        return batches, down_scale_ratio, pad_size

    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]

    asp_ratio = h / w
    down_scale_ratio = h / tgt_size

    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None

    if transpose:
        img = einops.rearrange(img, 'h w c -> w h c')

    pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w

    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    batches, down_scale_ratio, pad_size = _patch2batches(patch_list, p_num, transpose)

    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)
        for d, m in zip(db, mask):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)

    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask


def check_bubble_color(image):
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    gray_image = gray_image[..., np.newaxis]
    color_distance = np.sum((image - gray_image) ** 2, axis=-1)
    return np.sum(color_distance > 100) > 10


def is_ignore_bubble(region_img, ignore_bubble=0):
    if ignore_bubble < 1 or ignore_bubble > 50:
        return False
    _, binary_raw_mask = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)
    height, width = binary_raw_mask.shape[:2]
    total = 0
    val0 = 0
    val0 += sum(binary_raw_mask[0:2, 0:width].ravel() == 0)
    total += binary_raw_mask[0:2, 0:width].size
    val0 += sum(binary_raw_mask[height - 2:height, 0:width].ravel() == 0)
    total += binary_raw_mask[height - 2:height, 0:width].size
    val0 += sum(binary_raw_mask[2:height - 2, 0:2].ravel() == 0)
    total += binary_raw_mask[2:height - 2, 0:2].size
    val0 += sum(binary_raw_mask[2:height - 2, width - 2:width].ravel() == 0)
    total += binary_raw_mask[2:height - 2, width - 2:width].size
    ratio = round(val0 / total, 6) * 100
    if ignore_bubble <= ratio <= (100 - ignore_bubble):
        return True
    if check_bubble_color(region_img):
        return True
    return False


class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, val=None):
        if val is not None:
            self.sum += val
            self.count += 1
        return self.sum / self.count if self.count > 0 else 0
