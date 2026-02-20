import functools
from typing import List, Tuple
from functools import cached_property

import cv2
import numpy as np
import py3langid as langid
from shapely.geometry import Polygon, MultiPoint

from .geometry import (
    sort_points,
    point_distance,
    point_to_segment_distance,
    euclidean_distance,
    rotate_polygons,
)


def _color_difference(rgb1, rgb2) -> float:
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = (
        cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32)
        - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float32)
    )
    diff[..., 0] *= 0.392
    return np.linalg.norm(diff, axis=2).item()


def _is_valuable_char(ch):
    import unicodedata
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


def _is_right_to_left_char(ch):
    return (
        '\u0600' <= ch <= '\u06FF' or
        '\u0750' <= ch <= '\u077F' or
        '\u08A0' <= ch <= '\u08FF' or
        '\uFB50' <= ch <= '\uFDFF' or
        '\uFE70' <= ch <= '\uFEFF' or
        '\U00010E60' <= ch <= '\U00010E7F' or
        '\U0001EE00' <= ch <= '\U0001EEFF'
    )


class BBox:
    def __init__(self, x, y, w, h, text='', prob=0.0,
                 fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b

    def to_points(self):
        tl = np.array([self.x, self.y])
        tr = np.array([self.x + self.w, self.y])
        br = np.array([self.x + self.w, self.y + self.h])
        bl = np.array([self.x, self.y + self.h])
        return tl, tr, br, bl

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int32)


class TextLine:
    def __init__(self, pts: np.ndarray, text: str = '', prob: float = 0.0,
                 fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.pts, is_vertical = sort_points(pts)
        self.direction = 'v' if is_vertical else 'h'
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b
        self.assigned_direction: str = None
        self.textlines: List['TextLine'] = []

    @functools.cached_property
    def structure(self) -> List[np.ndarray]:
        p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
        p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
        p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
        p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
        return [p1, p2, p3, p4]

    @functools.cached_property
    def valid(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) * 180 / np.pi
        return abs(angle - 90) < 10

    @property
    def fg_colors(self):
        return np.array([self.fg_r, self.fg_g, self.fg_b])

    @property
    def bg_colors(self):
        return np.array([self.bg_r, self.bg_g, self.bg_b])

    @functools.cached_property
    def aspect_ratio(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return np.linalg.norm(v2) / np.linalg.norm(v1)

    @functools.cached_property
    def font_size(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return min(np.linalg.norm(v2), np.linalg.norm(v1))

    def width(self) -> int:
        return self.aabb.w

    def height(self) -> int:
        return self.aabb.h

    @functools.cached_property
    def xyxy(self):
        return self.aabb.x, self.aabb.y, self.aabb.x + self.aabb.w, self.aabb.y + self.aabb.h

    def clip(self, width, height):
        self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
        self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

    @functools.cached_property
    def aabb(self) -> BBox:
        kq = self.pts
        max_coord = np.max(kq, axis=0)
        min_coord = np.min(kq, axis=0)
        return BBox(
            min_coord[0], min_coord[1],
            max_coord[0] - min_coord[0], max_coord[1] - min_coord[1],
            self.text, self.prob,
            self.fg_r, self.fg_g, self.fg_b,
            self.bg_r, self.bg_g, self.bg_b,
        )

    def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v_vec = l1b - l1a
        h_vec = l2b - l2a
        ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

        src_pts = self.pts.astype(np.int64).copy()
        im_h, im_w = img.shape[:2]
        x1, y1 = src_pts[:, 0].min(), src_pts[:, 1].min()
        x2, y2 = src_pts[:, 0].max(), src_pts[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_cropped = img[y1:y2, x1:x2]
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1

        self.assigned_direction = direction
        if direction == 'h':
            h = max(int(textheight), 2)
            w = max(int(round(textheight / ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return cv2.warpPerspective(img_cropped, M, (w, h))
        elif direction == 'v':
            w = max(int(textheight), 2)
            h = max(int(round(textheight * ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_cropped, M, (w, h))
            return cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @functools.cached_property
    def is_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        return abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2

    @functools.cached_property
    def is_approximate_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        return (
            abs(np.dot(uv1, e1)) < 0.05 or abs(np.dot(uv1, e2)) < 0.05
            or abs(np.dot(uv2, e1)) < 0.05 or abs(np.dot(uv2, e2)) < 0.05
        )

    @functools.cached_property
    def cosangle(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        return np.dot(v1 / np.linalg.norm(v1), np.array([1, 0]))

    @functools.cached_property
    def angle(self) -> float:
        return np.fmod(np.arccos(self.cosangle) + np.pi, np.pi)

    @functools.cached_property
    def centroid(self) -> np.ndarray:
        return np.average(self.pts, axis=0)

    def distance_to_point(self, p: np.ndarray) -> float:
        d = 1.0e20
        for i in range(4):
            d = min(d, point_distance(p, self.pts[i]))
            d = min(d, point_to_segment_distance(p, self.pts[i], self.pts[(i + 1) % 4]))
        return d

    @functools.cached_property
    def polygon(self) -> Polygon:
        return MultiPoint([tuple(self.pts[i]) for i in range(4)]).convex_hull

    @functools.cached_property
    def area(self) -> float:
        return self.polygon.area

    def poly_distance(self, other) -> float:
        return self.polygon.distance(other.polygon)

    def distance(self, other, rho=0.5) -> float:
        fs = max(self.font_size, other.font_size)
        if self.assigned_direction == 'h':
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[3]), tuple(other.pts[0]), tuple(other.pts[3])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[1]), tuple(other.pts[2]), tuple(other.pts[1])]).convex_hull
            poly3 = MultiPoint([
                tuple(self.structure[0]), tuple(self.structure[1]),
                tuple(other.structure[0]), tuple(other.structure[1]),
            ]).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            dist3 = poly3.area / fs
            pattern = 'h_left'
            if dist1 < fs * rho:
                pattern = 'h_left'
            if dist2 < fs * rho and dist2 < dist1:
                pattern = 'h_right'
            if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2:
                pattern = 'h_middle'
            if pattern == 'h_left':
                return euclidean_distance(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            elif pattern == 'h_right':
                return euclidean_distance(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
            else:
                return euclidean_distance(self.structure[0][0], self.structure[0][1], other.structure[0][0], other.structure[0][1])
        else:
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(other.pts[0]), tuple(other.pts[1])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[3]), tuple(other.pts[2]), tuple(other.pts[3])]).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            pattern = 'v_top'
            if dist1 < fs * rho:
                pattern = 'v_top'
            if dist2 < fs * rho and dist2 < dist1:
                pattern = 'v_bottom'
            if pattern == 'v_top':
                return euclidean_distance(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            else:
                return euclidean_distance(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

    def copy(self, new_pts: np.ndarray):
        return TextLine(new_pts, self.text, self.prob, *self.fg_colors, *self.bg_colors)


LANGUAGE_ORIENTATION_PRESETS = {
    'CHS': 'auto', 'CHT': 'auto', 'CSY': 'h', 'NLD': 'h', 'ENG': 'h',
    'FRA': 'h', 'DEU': 'h', 'HUN': 'h', 'ITA': 'h', 'JPN': 'auto',
    'KOR': 'h', 'POL': 'h', 'PTB': 'h', 'ROM': 'h', 'RUS': 'h',
    'ESP': 'h', 'TRK': 'h', 'UKR': 'h', 'VIN': 'h', 'ARA': 'hr', 'FIL': 'h',
}


class TextRegion:
    def __init__(
        self,
        lines: List,
        texts: List[str] = None,
        language: str = 'unknown',
        font_size: float = -1,
        angle: float = 0,
        translation: str = "",
        fg_color: Tuple = (0, 0, 0),
        bg_color: Tuple = (0, 0, 0),
        line_spacing=1.0,
        letter_spacing=1.0,
        font_family: str = "",
        bold: bool = False,
        underline: bool = False,
        italic: bool = False,
        direction: str = 'auto',
        alignment: str = 'auto',
        rich_text: str = "",
        _bounding_rect=None,
        default_stroke_width=0.2,
        font_weight=50,
        source_lang: str = "",
        target_lang: str = "",
        opacity: float = 1.0,
        shadow_radius: float = 0.0,
        shadow_strength: float = 1.0,
        shadow_color: Tuple = (0, 0, 0),
        shadow_offset=None,
        prob: float = 1,
        **kwargs,
    ) -> None:
        if shadow_offset is None:
            shadow_offset = [0, 0]
        self.lines = np.array(lines, dtype=np.int32)
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.texts = texts if texts is not None else []
        self.text = texts[0] if texts else ""
        if self.text and texts and len(texts) > 1:
            for txt in texts[1:]:
                first_cjk = '\u3000' <= self.text[-1] <= '\u9fff'
                second_cjk = txt and ('\u3000' <= txt[0] <= '\u9fff')
                if first_cjk or second_cjk:
                    self.text += txt
                else:
                    self.text += ' ' + txt
        self.prob = prob
        self.translation = translation
        self.fg_colors = fg_color
        self.bg_colors = bg_color
        self.font_family = font_family
        self.bold = bold
        self.underline = underline
        self.italic = italic
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self._alignment = alignment
        self._source_lang = source_lang
        self.target_lang = target_lang
        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.adjust_bg_color = True
        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset

    @cached_property
    def xyxy(self):
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return np.array([x1, y1, x2, y2]).astype(np.int32)

    @cached_property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.int32)

    @cached_property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    @cached_property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @cached_property
    def unrotated_min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.clip(0).reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def polygon_aspect_ratio(self) -> float:
        polygons = self.unrotated_polygons.reshape(-1, 4, 2)
        middle_pts = (polygons[:, [1, 2, 3, 0]] + polygons) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        return np.mean(norm_h / norm_v)

    @cached_property
    def aspect_ratio(self) -> float:
        middle_pts = (self.min_rect[:, [1, 2, 3, 0]] + self.min_rect) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3])
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0])
        return norm_h / norm_v

    @property
    def polygon_object(self) -> Polygon:
        min_rect = self.min_rect[0]
        return MultiPoint([tuple(min_rect[i]) for i in range(4)]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    @property
    def real_area(self) -> float:
        lines = self.lines.reshape((-1, 2))
        return MultiPoint([tuple(l) for l in lines]).convex_hull.area

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    @property
    def src_is_vertical(self):
        if len(self.lines) == 0:
            return False
        polygons = self.unrotated_polygons.reshape(-1, 4, 2)
        middle_pts = (polygons[:, [1, 2, 3, 0]] + polygons) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        return np.mean(norm_h / norm_v) < 1

    def get_transformed_region(self, img: np.ndarray, line_idx: int, textheight: int, maxwidth=None) -> np.ndarray:
        im_h, im_w = img.shape[:2]
        line = np.round(np.array(self.lines[line_idx])).astype(np.int64)
        x1, y1 = line[:, 0].min(), line[:, 1].min()
        x2, y2 = line[:, 0].max(), line[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_cropped = img[y1:y2, x1:x2]

        direction = 'v' if self.src_is_vertical else 'h'
        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]
        vec_h = middle_pnt[1] - middle_pnt[3]
        norm_v = np.linalg.norm(vec_v)
        norm_h = np.linalg.norm(vec_h)

        if textheight is None:
            textheight = int(norm_v) if direction == 'h' else int(norm_h)

        if norm_v <= 0 or norm_h <= 0:
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == 'h':
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_cropped, M, (w, h))
        elif direction == 'v':
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_cropped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if maxwidth is not None:
            h, w = region.shape[:2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))
        return region

    @property
    def source_lang(self):
        if not self._source_lang:
            self._source_lang = langid.classify(self.text)[0]
        return self._source_lang

    def set_font_colors(self, fg_colors, bg_colors):
        self.fg_colors = np.array(fg_colors)
        self.bg_colors = np.array(bg_colors)

    def update_font_colors(self, fg_colors: np.ndarray, bg_colors: np.ndarray):
        nlines = len(self)
        if nlines > 0:
            self.fg_colors += fg_colors / nlines
            self.bg_colors += bg_colors / nlines

    def get_font_colors(self, bgr=False):
        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)
        if bgr:
            frgb = frgb[::-1]
            brgb = brgb[::-1]
        if self.adjust_bg_color:
            fg_avg = np.mean(frgb)
            if _color_difference(frgb, brgb) < 30:
                brgb = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
        return frgb, brgb

    @property
    def direction(self):
        if self._direction not in ('h', 'v', 'hr', 'vr'):
            d = LANGUAGE_ORIENTATION_PRESETS.get(self.target_lang)
            if d in ('h', 'v', 'hr', 'vr'):
                return d
            if len(self.lines) > 0:
                max_area = 0
                largest_box_aspect_ratio = 1
                for line in self.lines:
                    line_polygon = Polygon(line)
                    area = line_polygon.area
                    if area > max_area:
                        max_area = area
                        x_coords = line[:, 0]
                        y_coords = line[:, 1]
                        w = np.max(x_coords) - np.min(x_coords)
                        h = np.max(y_coords) - np.min(y_coords)
                        largest_box_aspect_ratio = w / h if h > 0 else 1
                return 'v' if largest_box_aspect_ratio < 1 else 'h'
            else:
                return 'v' if self.aspect_ratio < 1 else 'h'
        return self._direction

    @property
    def vertical(self):
        return self.direction.startswith('v')

    @property
    def horizontal(self):
        return self.direction.startswith('h')

    @property
    def alignment(self):
        if self._alignment in ('left', 'center', 'right'):
            return self._alignment
        if len(self.lines) == 1:
            return 'center'
        if self.direction == 'h':
            return 'center'
        elif self.direction == 'hr':
            return 'right'
        return 'left'

    @property
    def stroke_width(self):
        diff = _color_difference(*self.get_font_colors())
        if diff > 15:
            return self.default_stroke_width
        return 0

    def normalized_width_list(self) -> List[float]:
        polygons = self.unrotated_polygons
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).sum())
        width_list = np.array(width_list)
        width_list = width_list / np.sum(width_list)
        return width_list.tolist()
