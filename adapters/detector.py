from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from engine.detection import TextDetector


class MangaTranslatorDetector:

    def __init__(
        self,
        detection_size: int = 2048,
        text_threshold: float = 0.5,
        box_threshold: float = 0.7,
        unclip_ratio: float = 2.3,
        device: str = "cpu",
    ):
        self._engine = TextDetector()
        self._detection_size = detection_size
        self._text_threshold = text_threshold
        self._box_threshold = box_threshold
        self._unclip_ratio = unclip_ratio
        self._device = device

    async def prepare(self) -> None:
        await self._engine.load(device=self._device)

    async def detect(
        self,
        img_rgb: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[list, Optional[np.ndarray], Optional[np.ndarray]]:
        return await self._engine.infer(
            img_rgb,
            self._detection_size,
            self._text_threshold,
            self._box_threshold,
            self._unclip_ratio,
            verbose=verbose,
        )
