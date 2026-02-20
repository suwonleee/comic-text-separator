# GPL-3.0
import os
import itertools
from typing import List, Union
from collections import Counter

import cv2
import numpy as np
import torch
import einops
import networkx as nx

from .nets.ocr_net import OCR
from .model_manager import WeightManager
from .types import TextLine, TextRegion
from .geometry import can_merge_textlines
from .image_utils import AvgMeter, chunks


class OcrRecognizer(WeightManager):
    _MODEL_SUB_DIR = 'ocr'
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt',
            'hash': '29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d',
        },
        'dict': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt',
            'hash': 'f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a',
        },
    }

    async def _load(self, device: str):
        with open(self._get_file_path('alphabet-all-v7.txt'), 'r', encoding='utf-8') as fp:
            dictionary = [s[:-1] for s in fp.readlines()]
        self.model = OCR(dictionary, 768)
        sd = torch.load(self._get_file_path('ocr_ar_48px.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device
        self.use_gpu = device in ('cuda', 'mps')
        if self.use_gpu:
            self.model = self.model.to(device)

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, textlines: List[TextLine],
                     verbose: bool = False) -> List[TextLine]:
        text_height = 48
        max_chunk_size = 16

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]
        out_regions = []

        perm = range(len(region_imgs))
        is_textlines = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], TextLine):
            perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])
            is_textlines = True

        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                W = region_imgs[idx].shape[1]
                region[i, :, :W, :] = region_imgs[idx]

            image_tensor = (torch.from_numpy(region).float() - 127.5) / 127.5
            image_tensor = einops.rearrange(image_tensor, 'N H W C -> N C H W')
            if self.use_gpu:
                image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                ret = self.model.infer_beam_batch_tensor(image_tensor, widths, beams_k=5, max_seq_length=255)

            for i, (pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) in enumerate(ret):
                if prob < 0.2:
                    continue
                has_fg = (fg_ind_pred[:, 1] > fg_ind_pred[:, 0])
                has_bg = (bg_ind_pred[:, 1] > bg_ind_pred[:, 0])
                seq = []
                fr, fg, fb = AvgMeter(), AvgMeter(), AvgMeter()
                br, bg_, bb = AvgMeter(), AvgMeter(), AvgMeter()
                for chid, c_fg, c_bg, h_fg, h_bg in zip(pred_chars_index, fg_pred, bg_pred, has_fg, has_bg):
                    ch = self.model.dictionary[chid]
                    if ch == '<S>':
                        continue
                    if ch == '</S>':
                        break
                    if ch == '<SP>':
                        ch = ' '
                    seq.append(ch)
                    if h_fg.item():
                        fr(int(c_fg[0] * 255))
                        fg(int(c_fg[1] * 255))
                        fb(int(c_fg[2] * 255))
                    if h_bg.item():
                        br(int(c_bg[0] * 255))
                        bg_(int(c_bg[1] * 255))
                        bb(int(c_bg[2] * 255))
                    else:
                        br(int(c_fg[0] * 255))
                        bg_(int(c_fg[1] * 255))
                        bb(int(c_fg[2] * 255))

                txt = ''.join(seq)
                fr_v = min(max(int(fr()), 0), 255)
                fg_v = min(max(int(fg()), 0), 255)
                fb_v = min(max(int(fb()), 0), 255)
                br_v = min(max(int(br()), 0), 255)
                bg_v = min(max(int(bg_()), 0), 255)
                bb_v = min(max(int(bb()), 0), 255)

                self.logger.info(f'prob: {prob} {txt} fg: ({fr_v}, {fg_v}, {fb_v}) bg: ({br_v}, {bg_v}, {bb_v})')
                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, TextLine):
                    cur_region.text = txt
                    cur_region.prob = prob
                    cur_region.fg_r = fr_v
                    cur_region.fg_g = fg_v
                    cur_region.fg_b = fb_v
                    cur_region.bg_r = br_v
                    cur_region.bg_g = bg_v
                    cur_region.bg_b = bb_v
                else:
                    cur_region.text.append(txt)
                    cur_region.update_font_colors(np.array([fr_v, fg_v, fb_v]), np.array([br_v, bg_v, bb_v]))
                out_regions.append(cur_region)

        if is_textlines:
            return out_regions
        return textlines

    @staticmethod
    def _generate_text_direction(bboxes: List[Union[TextLine, TextRegion]]):
        if len(bboxes) > 0:
            if isinstance(bboxes[0], TextRegion):
                for blk in bboxes:
                    for line_idx in range(len(blk.lines)):
                        yield blk, line_idx
            else:
                G = nx.Graph()
                for i, box in enumerate(bboxes):
                    G.add_node(i, box=box)
                for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
                    if can_merge_textlines(ubox, vbox, aspect_ratio_tol=1):
                        G.add_edge(u, v)
                for node_set in nx.algorithms.components.connected_components(G):
                    nodes = list(node_set)
                    dirs = [bboxes[i].direction for i in nodes]
                    majority_dir = Counter(dirs).most_common(1)[0][0]
                    if majority_dir == 'h':
                        nodes = sorted(nodes, key=lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
                    elif majority_dir == 'v':
                        nodes = sorted(nodes, key=lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
                    for node in nodes:
                        yield bboxes[node], majority_dir
