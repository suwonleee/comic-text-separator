from __future__ import annotations

import logging

import numpy as np

from engine.inpainting import ImageInpainter, MaskRefiner

logger = logging.getLogger(__name__)


class MangaTranslatorInpainter:

    def __init__(
        self,
        inpainting_size: int = 2048,
        mask_dilation_offset: int = 20,
        device: str = "cpu",
    ):
        self._engine = ImageInpainter()
        self._inpainting_size = inpainting_size
        self._mask_dilation_offset = mask_dilation_offset
        self._device = device

    async def prepare(self) -> None:
        await self._engine.load(device=self._device)

    async def refine_mask(
        self,
        text_regions: list,
        img_rgb: np.ndarray,
        mask_raw: np.ndarray,
    ) -> np.ndarray:
        return MaskRefiner.refine(
            text_regions,
            img_rgb,
            mask_raw,
            dilation_offset=self._mask_dilation_offset,
            kernel_size=3,
        )

    async def inpaint(
        self,
        img_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        return await self._engine.infer(
            img_rgb,
            mask,
            inpainting_size=self._inpainting_size,
        )
