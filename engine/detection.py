# GPL-3.0
import numpy as np
import cv2
import torch
import einops
from typing import List, Tuple
from collections import Counter

import pyclipper
from shapely.geometry import Polygon

from .nets.dbnet import DBNetModel
from .model_manager import WeightManager
from .types import TextLine
from .image_utils import det_rearrange_forward, resize_aspect_ratio


_GLOBAL_MODEL = None


def _batch_forward(batch: np.ndarray, device: str):
    global _GLOBAL_MODEL
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, 'n h w c -> n c h w')
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = _GLOBAL_MODEL(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask


class DBNetPostProcessor:
    def __init__(self, thresh=0.6, box_thresh=0.8, max_candidates=1000, unclip_ratio=2.2):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch, scores_batch = [], []
        batch_size = pred.size(0) if isinstance(pred, torch.Tensor) else pred.shape[0]
        for bi in range(batch_size):
            height, width = batch['shape'][bi]
            boxes, scores = self._polygons_from_bitmap(pred[bi], segmentation[bi], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def _polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap.cpu().numpy() if isinstance(_bitmap, torch.Tensor) else _bitmap
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes, scores = [], []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self._box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self._unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self._get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


    @staticmethod
    def _unclip(box, unclip_ratio=1.8):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance))

    @staticmethod
    def _get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        idx1 = 0 if points[1][1] > points[0][1] else 1
        idx4 = 1 if points[1][1] > points[0][1] else 0
        idx2 = 2 if points[3][1] > points[2][1] else 3
        idx3 = 3 if points[3][1] > points[2][1] else 2
        return [points[idx1], points[idx2], points[idx3], points[idx4]], min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class TextDetector(WeightManager):
    _MODEL_SUB_DIR = 'detection'
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt',
            'hash': '67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e',
            'file': '.',
        },
    }

    async def _load(self, device: str):
        self.model = DBNetModel()
        sd = torch.load(self._get_file_path('detect-20241225.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        if device == 'cuda' or device == 'mps':
            self.model = self.model.to(self.device)
        global _GLOBAL_MODEL
        _GLOBAL_MODEL = self.model

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float,
                     box_threshold: float, unclip_ratio: float,
                     invert: bool = False, gamma_correct: bool = False,
                     rotate: bool = False, auto_rotate: bool = False,
                     verbose: bool = False):
        orig_image = image.copy()
        img_h, img_w = image.shape[:2]
        minimum_image_size = 400
        add_border = min(img_w, img_h) < minimum_image_size
        if rotate:
            image = np.rot90(image, k=-1)
        if add_border:
            image = self._add_border(image, minimum_image_size)
        if invert:
            image = cv2.bitwise_not(image)
        if gamma_correct:
            image = self._apply_gamma(image)

        textlines, raw_mask, mask = await self._run_detection(image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose)
        textlines = list(filter(lambda x: x.area > 1, textlines))

        if add_border:
            textlines, raw_mask, mask = self._remove_border(image, img_w, img_h, textlines, raw_mask, mask)
        if auto_rotate:
            if len(textlines) > 0:
                orientations = ['h' if t.aspect_ratio > 1 else 'v' for t in textlines]
                majority = Counter(orientations).most_common(1)[0][0]
            else:
                majority = 'h'
            if majority == 'h':
                return await self._infer(orig_image, detect_size, text_threshold, box_threshold,
                                         unclip_ratio, invert, gamma_correct, rotate=(not rotate),
                                         auto_rotate=False, verbose=verbose)
        if rotate:
            textlines, raw_mask, mask = self._remove_rotation(textlines, raw_mask, mask, img_w, img_h)

        return textlines, raw_mask, mask

    async def _run_detection(self, image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose):
        db, mask = det_rearrange_forward(image, _batch_forward, detect_size, 4, device=self.device, verbose=verbose)
        pad_h = pad_w = 0
        if db is None:
            img_resized, target_ratio, _, pad_w, pad_h = resize_aspect_ratio(
                cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio=1)
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio
            db, mask = _batch_forward([img_resized], self.device)
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1

        self.logger.info(f'Detection resolution: {img_resized_w}x{img_resized_h}')
        mask = mask[0, 0, :, :]
        det = DBNetPostProcessor(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        boxes, scores = det({'shape': [(img_resized_h, img_resized_w)]}, db)
        boxes, scores = boxes[0], scores[0]
        # Polygon N-point → 4-point minAreaRect for TextLine compatibility
        polys = []
        valid_scores = []
        for poly_pts, score in zip(boxes, scores):
            pts = np.array(poly_pts, dtype=np.float64)
            pts[:, 0] *= ratio_w
            pts[:, 1] *= ratio_h
            rect = cv2.minAreaRect(pts.astype(np.float32).reshape(-1, 1, 2))
            box_4pt = cv2.boxPoints(rect).astype(np.int64)
            polys.append(box_4pt)
            valid_scores.append(score)
        scores = valid_scores
        textlines = [TextLine(pts.astype(int) if isinstance(pts, np.ndarray) else np.array(pts, dtype=int), '', score)
                     for pts, score in zip(polys, scores)]
        textlines = list(filter(lambda q: q.area > 16, textlines))

        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if pad_h > 0:
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0:
            mask_resized = mask_resized[:, :-pad_w]
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
        return textlines, raw_mask, None

    @staticmethod
    def _add_border(image, target_side_length):
        old_h, old_w = image.shape[:2]
        new_w = new_h = max(old_w, old_h, target_side_length)
        new_image = np.zeros([new_h, new_w, 3], dtype=np.uint8)
        new_image[:old_h, :old_w] = image
        return new_image

    @staticmethod
    def _remove_border(image, old_w, old_h, textlines, raw_mask, mask):
        new_h, new_w = image.shape[:2]
        raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        raw_mask = raw_mask[:old_h, :old_w]
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = mask[:old_h, :old_w]
        new_textlines = []
        for txtln in textlines:
            if txtln.xyxy[0] >= old_w and txtln.xyxy[1] >= old_h:
                continue
            points = txtln.pts
            points[:, 0] = np.clip(points[:, 0], 0, old_w)
            points[:, 1] = np.clip(points[:, 1], 0, old_h)
            new_textlines.append(TextLine(points, txtln.text, txtln.prob))
        return new_textlines, raw_mask, mask

    @staticmethod
    def _remove_rotation(textlines, raw_mask, mask, img_w, img_h):
        raw_mask = np.ascontiguousarray(np.rot90(raw_mask))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask).astype(np.uint8))
        for i, txtln in enumerate(textlines):
            rotated_pts = txtln.pts[:, [1, 0]]
            rotated_pts[:, 1] = -rotated_pts[:, 1] + img_h
            textlines[i] = TextLine(rotated_pts, txtln.text, txtln.prob)
        return textlines, raw_mask, mask

    @staticmethod
    def _apply_gamma(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        gamma = np.log(0.5 * 255) / np.log(mean)
        return np.power(image, gamma).clip(0, 255).astype(np.uint8)
