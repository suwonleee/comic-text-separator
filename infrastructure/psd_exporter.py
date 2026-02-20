from __future__ import annotations

import logging
import os
import platform
import glob
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from psd_tools import PSDImage
from psd_tools.constants import Compression

from domain.models import ExtractionResult, TextRegionData

logger = logging.getLogger(__name__)

_CJK_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

_DARWIN_FONTS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

_LINUX_FONT_GLOBS = [
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansKR*.otf",
    "/usr/share/fonts/**/malgun*.ttf",
]

_LINUX_FONT_FALLBACK = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"


def _find_cjk_font_path() -> Optional[str]:
    system = platform.system()

    if system == "Darwin":
        candidates = _DARWIN_FONTS
    elif system == "Linux":
        candidates = []
        for pattern in _LINUX_FONT_GLOBS:
            candidates.extend(glob.glob(pattern, recursive=True))
        candidates.append(_LINUX_FONT_FALLBACK)
    elif system == "Windows":
        fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        candidates = [
            os.path.join(fonts_dir, "malgun.ttf"),
            os.path.join(fonts_dir, "msgothic.ttc"),
            os.path.join(fonts_dir, "simsun.ttc"),
        ]
    else:
        candidates = []

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def _get_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[size]

    path = _find_cjk_font_path()
    if path:
        font = ImageFont.truetype(path, size)
        logger.debug(f"Using CJK font: {path} @ {size}px")
    else:
        logger.warning(
            "No CJK font found. Text will render with default font. "
            "Install Noto Sans CJK for best results."
        )
        font = ImageFont.load_default()

    _CJK_FONT_CACHE[size] = font
    return font


_LAYER_PADDING = 4


def _wrap_horizontal_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]
        if line_width > max_width and current_line:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines


def _render_text_layer_image(region: TextRegionData) -> Image.Image:
    font_size = max(region.font_size, 10)
    font = _get_cjk_font(font_size)

    display_text = region.text_corrected if region.text_corrected else region.text

    region_w = max(region.width, 20)
    region_h = max(region.height, 20)

    if region.direction == "v":
        sample_bbox = font.getbbox("가")
        char_h = sample_bbox[3] - sample_bbox[1]
        char_w = sample_bbox[2] - sample_bbox[0]
        spacing = int(char_h * 0.15)
        col_width = int(char_w * 1.3)

        cols = 1
        y = 0
        for char in display_text:
            if char == "\n" or y + char_h > region_h:
                cols += 1
                y = 0
                if char == "\n":
                    continue
            y += char_h + spacing

        needed_w = cols * col_width + char_w
        canvas_w = max(region_w, needed_w) + _LAYER_PADDING * 2
        canvas_h = region_h + _LAYER_PADDING * 2
    else:
        lines = _wrap_horizontal_lines(display_text, font, region_w)
        total_h = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_height = bbox[3] - bbox[1]
            total_h += int(line_height * 1.2)

        canvas_w = region_w + _LAYER_PADDING * 2
        canvas_h = max(region_h, total_h) + _LAYER_PADDING * 2

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r, g, b = region.fg_color
    fill = (r, g, b, 255)

    if region.direction == "v":
        _draw_vertical_text(draw, display_text, font, fill, _LAYER_PADDING, canvas_w, canvas_h)
    else:
        _draw_horizontal_text(draw, display_text, font, fill, _LAYER_PADDING, canvas_w, canvas_h)

    return img


def _draw_horizontal_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    padding: int,
    canvas_w: int,
    canvas_h: int,
) -> None:
    max_width = canvas_w - padding * 2
    lines: list[str] = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]
        if line_width > max_width and current_line:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    y_offset = padding
    for line in lines:
        draw.text((padding, y_offset), line, font=font, fill=fill)
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y_offset += int(line_height * 1.2)


def _draw_vertical_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    padding: int,
    canvas_w: int,
    canvas_h: int,
) -> None:
    max_height = canvas_h - padding * 2

    sample_bbox = font.getbbox("가")
    char_h = sample_bbox[3] - sample_bbox[1]
    char_w = sample_bbox[2] - sample_bbox[0]
    spacing = int(char_h * 0.15)

    x = canvas_w - padding - char_w
    y = padding

    for char in text:
        if char == "\n" or y + char_h > max_height + padding:
            x -= int(char_w * 1.3)
            y = padding
            if char == "\n":
                continue

        draw.text((x, y), char, font=font, fill=fill)
        y += char_h + spacing


# PSD Pascal strings use mac_roman — CJK chars cause UnicodeEncodeError
def _make_safe_layer_name(index: int, text: str) -> str:
    ascii_preview = text[:20].replace("\n", " ").encode("ascii", errors="replace").decode("ascii")
    return f"Text {index}: {ascii_preview}"


def save_psd(
    result: ExtractionResult,
    output_path: str,
    inpainted_img: Optional[np.ndarray] = None,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if not os.path.isfile(result.image_path):
        raise FileNotFoundError(f"Original image not found: {result.image_path}")

    original = Image.open(result.image_path).convert("RGBA")
    img_w, img_h = original.size

    logger.info(f"Creating PSD: {img_w}x{img_h}, {len(result.regions)} text region(s)")

    psd = PSDImage.new(mode="RGBA", size=(img_w, img_h))

    psd.create_pixel_layer(
        original,
        name="Original Image",
        top=0,
        left=0,
        compression=Compression.RLE,
    )

    if inpainted_img is not None:
        inpainted_pil = Image.fromarray(inpainted_img).convert("RGBA")
        psd.create_pixel_layer(
            inpainted_pil,
            name="Inpainted (Text Removed)",
            top=0,
            left=0,
            compression=Compression.RLE,
        )
        logger.debug("  Added inpainted layer")

    if not result.regions:
        psd.save(output_path)
        return output_path

    group = psd.create_group(name="Text Layers")

    for i, region in enumerate(result.regions):
        text_img = _render_text_layer_image(region)

        layer_name = _make_safe_layer_name(i + 1, region.text)

        top = max(region.y - _LAYER_PADDING, 0)
        left = max(region.x - _LAYER_PADDING, 0)

        layer = psd.create_pixel_layer(
            text_img,
            name=layer_name,
            top=top,
            left=left,
            compression=Compression.RLE,
        )
        group.append(layer)
        logger.debug(f"  Added layer: {layer_name} @ ({left}, {top})")

    psd.save(output_path)
    file_size = os.path.getsize(output_path)
    logger.info(f"  Saved PSD: {output_path} ({len(result.regions)} layers, {file_size / 1024:.0f} KB)")
    return output_path
