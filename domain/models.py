"""
Clean data models for text extraction results.
Decoupled from manga-image-translator internals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


@dataclass
class TextRegionData:
    """A single detected & OCR'd text region."""

    # ── content ──
    text: str
    """OCR-recognized text content."""

    # ── geometry ──
    x: int
    y: int
    width: int
    height: int
    angle: float = 0.0
    """Rotation angle in degrees."""

    # ── polygon (original quadrilateral points) ──
    polygon: List[List[int]] = field(default_factory=list)
    """[[x1,y1],[x2,y2],[x3,y3],[x4,y4]] bounding quadrilateral."""

    # ── typography ──
    font_size: int = 20
    fg_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    bold: bool = False
    italic: bool = False
    direction: str = "h"
    """'h' = horizontal, 'v' = vertical, 'hr' = horizontal RTL."""
    alignment: str = "left"
    """'left', 'center', 'right'."""

    # ── metadata ──
    source_lang: str = ""
    confidence: float = 1.0
    line_spacing: float = 1.0
    letter_spacing: float = 0.0
    text_corrected: str = ""


@dataclass
class ExtractionResult:
    """Complete extraction result for one image."""

    image_path: str
    image_width: int
    image_height: int
    regions: List[TextRegionData] = field(default_factory=list)
    inpainted_image_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
