# GPL-3.0
import numpy as np
import cv2
import torch
from typing import List, Tuple

from shapely.geometry import Polygon
from tqdm import tqdm

from .nets.aot import AOTGenerator
from .model_manager import WeightManager
from .types import TextLine, TextRegion
from .image_utils import resize_keep_aspect, is_ignore_bubble

try:
    from pydensecrf.utils import unary_from_softmax
    import pydensecrf.densecrf as dcrf
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False


def _refine_mask_crf(rgbimg, rawmask):
    if not _HAS_CRF:
        return rawmask
    if len(rawmask.shape) == 2:
        rawmask = rawmask[:, :, None]
    mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
    mask_softmax = mask_softmax.astype(np.float32) / 255.0
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((2, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION)
    d.addPairwiseBilateral(sxy=23, srgb=7, rgbim=rgbimg, compat=20, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
    return np.array(res * 255, dtype=np.uint8)


def _extend_rect(x, y, w, h, max_x, max_y, extend_size):
    x1 = max(x - extend_size, 0)
    y1 = max(y - extend_size, 0)
    w1 = min(w + extend_size * 2, max_x - x1 - 1)
    h1 = min(h + extend_size * 2, max_y - y1 - 1)
    return x1, y1, w1, h1


class MaskRefiner:
    @staticmethod
    def refine(text_regions: List[TextRegion], raw_image: np.ndarray, raw_mask: np.ndarray,
               dilation_offset: int = 0, ignore_bubble: int = 0, kernel_size: int = 3) -> np.ndarray:
        scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)
        img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
        mask_resized[mask_resized > 0] = 255

        textlines = []
        for region in text_regions:
            for l in region.lines:
                textlines.append(TextLine(l * scale_factor, '', 0))

        final_mask = MaskRefiner._complete_mask(img_resized, mask_resized, textlines, dilation_offset=dilation_offset, kernel_size=kernel_size)
        if final_mask is None:
            final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.uint8)
        else:
            final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            final_mask[final_mask > 0] = 255

        if ignore_bubble < 1 or ignore_bubble > 50:
            return final_mask

        k_size = int(max(final_mask.shape) * 0.025)
        kern = np.ones((k_size, k_size), np.uint8)
        final_mask = cv2.dilate(final_mask, kern, iterations=1)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            temp_mask = np.zeros_like(final_mask)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
            textblock = cv2.bitwise_and(raw_image, raw_image, mask=temp_mask)
            if is_ignore_bubble(textblock, ignore_bubble):
                cv2.drawContours(final_mask, [cnt], -1, 0, -1)
        return final_mask

    @staticmethod
    def _complete_mask(img, mask, textlines, keep_threshold=1e-2, dilation_offset=0, kernel_size=3):
        bboxes = [txtln.aabb.xywh for txtln in textlines]
        polys = [Polygon(txtln.pts) for txtln in textlines]
        for (x, y, w, h) in bboxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (0), 1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        M = len(textlines)
        textline_ccs = [np.zeros_like(mask) for _ in range(M)]
        iinfo = np.iinfo(labels.dtype)
        textline_rects = np.full(shape=(M, 4), fill_value=[iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype=labels.dtype)
        ratio_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
        dist_mat = np.zeros(shape=(num_labels, M), dtype=np.float32)
        valid = False

        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] <= 9:
                continue
            x1 = stats[label, cv2.CC_STAT_LEFT]
            y1 = stats[label, cv2.CC_STAT_TOP]
            w1 = stats[label, cv2.CC_STAT_WIDTH]
            h1 = stats[label, cv2.CC_STAT_HEIGHT]
            area1 = stats[label, cv2.CC_STAT_AREA]
            cc_pts = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
            cc_poly = Polygon(cc_pts)

            for tl_idx in range(M):
                area2 = polys[tl_idx].area
                overlapping_area = polys[tl_idx].intersection(cc_poly).area
                ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
                dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)

            avg = np.argmax(ratio_mat[label])
            area2 = polys[avg].area
            if area1 >= area2:
                continue
            if ratio_mat[label, avg] <= keep_threshold:
                avg = np.argmin(dist_mat[label])
                unit = max(min([textlines[avg].font_size, w1, h1]), 10)
                if dist_mat[label, avg] >= 0.5 * unit:
                    continue

            textline_ccs[avg][y1:y1 + h1, x1:x1 + w1][labels[y1:y1 + h1, x1:x1 + w1] == label] = 255
            textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
            textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
            textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
            textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
            valid = True

        if not valid:
            return None

        textline_rects[:, 2] -= textline_rects[:, 0]
        textline_rects[:, 3] -= textline_rects[:, 1]

        final_mask = np.zeros_like(mask)
        img = cv2.bilateralFilter(img, 17, 80, 80)
        for i, cc in enumerate(tqdm(textline_ccs, '[mask]')):
            x1, y1, w1, h1 = textline_rects[i]
            text_size = min(w1, h1, textlines[i].font_size)
            x1, y1, w1, h1 = _extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], int(text_size * 0.1))
            dilate_size = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            cc_region = np.ascontiguousarray(cc[y1:y1 + h1, x1:x1 + w1])
            if cc_region.size == 0:
                continue
            img_region = np.ascontiguousarray(img[y1:y1 + h1, x1:x1 + w1])
            cc_region = _refine_mask_crf(img_region, cc_region)
            cc[y1:y1 + h1, x1:x1 + w1] = cc_region
            x2, y2, w2, h2 = _extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], -(-dilate_size // 2))
            cc[y2:y2 + h2, x2:x2 + w2] = cv2.dilate(cc[y2:y2 + h2, x2:x2 + w2], kern)
            final_mask[y2:y2 + h2, x2:x2 + w2] = cv2.bitwise_or(final_mask[y2:y2 + h2, x2:x2 + w2], cc[y2:y2 + h2, x2:x2 + w2])

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(final_mask, kern)


class ImageInpainter(WeightManager):
    _MODEL_SUB_DIR = 'inpainting'
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt',
            'hash': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'file': '.',
        },
    }

    async def _load(self, device: str):
        self.model = AOTGenerator()
        sd = torch.load(self._get_file_path('inpainting.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        if device.startswith('cuda') or device == 'mps':
            self.model.to(device)

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024,
                     verbose: bool = False) -> np.ndarray:
        img_original = np.copy(image)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        height, width = image.shape[:2]
        if max(image.shape[:2]) > inpainting_size:
            image = resize_keep_aspect(image, inpainting_size)
            mask = resize_keep_aspect(mask, inpainting_size)

        pad_size = 8
        h, w = image.shape[:2]
        new_h = h if h % pad_size == 0 else h + (pad_size - h % pad_size)
        new_w = w if w % pad_size == 0 else w + (pad_size - w % pad_size)
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.logger.info(f'Inpainting resolution: {new_w}x{new_h}')

        img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device.startswith('cuda') or self.device == 'mps':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)

        with torch.no_grad():
            img_torch *= (1 - mask_torch)
            img_inpainted_torch = self.model(img_torch, mask_torch)

        img_inpainted_torch = img_inpainted_torch.to(torch.float32)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)

        if new_h != height or new_w != width:
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation=cv2.INTER_LINEAR)

        return img_inpainted * mask_original + img_original * (1 - mask_original)
