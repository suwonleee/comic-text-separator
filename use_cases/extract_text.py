from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

from domain.models import ExtractionResult
from domain.ports import (
    DetectorPort,
    OcrPort,
    TextlineMergerPort,
    TextBlockMapperPort,
    InpainterPort,
    SpacingCorrectorPort,
)

logger = logging.getLogger(__name__)


class ExtractTextUseCase:

    def __init__(
        self,
        detector: DetectorPort,
        ocr: OcrPort,
        merger: TextlineMergerPort,
        mapper: TextBlockMapperPort,
        inpainter: Optional[InpainterPort] = None,
        spacing_corrector: Optional[SpacingCorrectorPort] = None,
    ):
        self._detector = detector
        self._ocr = ocr
        self._merger = merger
        self._mapper = mapper
        self._inpainter = inpainter
        self._spacing_corrector = spacing_corrector

    async def prepare(self) -> None:
        logger.info("Preparing detection model...")
        await self._detector.prepare()
        logger.info("Preparing OCR model...")
        await self._ocr.prepare()
        if self._inpainter:
            logger.info("Preparing inpainting model...")
            await self._inpainter.prepare()
        logger.info("Models ready.")

    async def execute(
        self, image_path: str, verbose: bool = False
    ) -> tuple[ExtractionResult, Optional[np.ndarray]]:
        logger.info(f"Processing: {image_path}")

        img_pil = Image.open(image_path).convert("RGB")
        img_rgb = np.array(img_pil)
        img_h, img_w = img_rgb.shape[:2]
        logger.info(f"  Image size: {img_w}x{img_h}")

        empty_result = ExtractionResult(
            image_path=os.path.abspath(image_path),
            image_width=img_w,
            image_height=img_h,
            regions=[],
        )

        logger.info("  Running text detection...")
        textlines, mask_raw, _mask = await self._detector.detect(img_rgb, verbose=verbose)

        if not textlines:
            logger.info("  No text detected.")
            return empty_result, None

        logger.info(f"  Detected {len(textlines)} textline(s)")

        logger.info("  Running OCR...")
        textlines = await self._ocr.recognize(img_rgb, textlines, verbose=verbose)

        textlines = [tl for tl in textlines if tl.text.strip()]
        logger.info(f"  OCR recognized {len(textlines)} textline(s) with text")

        if not textlines:
            return empty_result, None

        logger.info("  Merging textlines...")
        text_regions = await self._merger.merge(textlines, img_w, img_h, verbose=verbose)
        logger.info(f"  Merged into {len(text_regions)} text region(s)")

        result = self._mapper.map(
            image_path=image_path,
            img_width=img_w,
            img_height=img_h,
            text_regions=text_regions,
        )

        inpainted_img: Optional[np.ndarray] = None
        if self._inpainter and mask_raw is not None:
            logger.info("  Refining mask...")
            refined_mask = await self._inpainter.refine_mask(text_regions, img_rgb, mask_raw)
            logger.info("  Running inpainting...")
            inpainted_img = await self._inpainter.inpaint(img_rgb, refined_mask)
            logger.info("  Inpainting complete.")

        if self._spacing_corrector:
            logger.info("  Correcting Korean spacing...")
            result = self._spacing_corrector.correct(result)

        return result, inpainted_img
