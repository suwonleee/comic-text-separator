from __future__ import annotations

import os
from typing import List

from domain.models import TextRegionData, ExtractionResult


class TextBlockMapper:

    def map(
        self,
        image_path: str,
        img_width: int,
        img_height: int,
        text_regions: list,
    ) -> ExtractionResult:
        regions: List[TextRegionData] = []

        for region in text_regions:
            fg = region.fg_colors
            if isinstance(fg, (list, tuple)) and len(fg) >= 3:
                fg_color = (int(fg[0]), int(fg[1]), int(fg[2]))
            else:
                fg_color = (0, 0, 0)

            bg = region.bg_colors
            if isinstance(bg, (list, tuple)) and len(bg) >= 3:
                bg_color = (int(bg[0]), int(bg[1]), int(bg[2]))
            else:
                bg_color = (255, 255, 255)

            try:
                polygon = region.min_rect[0].tolist()
            except Exception:
                x1, y1, w, h = region.xywh
                polygon = [[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]]

            direction = "h"
            if hasattr(region, "_direction") and region._direction != "auto":
                direction = region._direction
            elif hasattr(region, "vertical") and region.vertical:
                direction = "v"

            alignment = "left"
            if hasattr(region, "_alignment") and region._alignment != "auto":
                alignment = region._alignment

            x, y, w, h = (
                int(region.xywh[0]),
                int(region.xywh[1]),
                int(region.xywh[2]),
                int(region.xywh[3]),
            )

            regions.append(TextRegionData(
                text=region.text.strip(),
                x=x,
                y=y,
                width=w,
                height=h,
                angle=float(region.angle),
                polygon=polygon,
                font_size=int(region.font_size) if region.font_size > 0 else 20,
                fg_color=fg_color,
                bg_color=bg_color,
                bold=getattr(region, "bold", False),
                italic=getattr(region, "italic", False),
                direction=direction,
                alignment=alignment,
                source_lang=getattr(region, "source_lang", ""),
                confidence=getattr(region, "prob", 1.0),
                line_spacing=getattr(region, "line_spacing", 1.0),
                letter_spacing=getattr(region, "letter_spacing", 0.0),
            ))

        return ExtractionResult(
            image_path=os.path.abspath(image_path),
            image_width=img_width,
            image_height=img_height,
            regions=regions,
        )
