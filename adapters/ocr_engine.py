from __future__ import annotations

import numpy as np

from engine.recognition import OcrRecognizer


class MangaTranslatorOcr:

    def __init__(
        self,
        device: str = "cpu",
    ):
        self._engine = OcrRecognizer()
        self._device = device

    async def prepare(self) -> None:
        await self._engine.load(device=self._device)

    async def recognize(
        self,
        img_rgb: np.ndarray,
        textlines: list,
        verbose: bool = False,
    ) -> list:
        return await self._engine.infer(img_rgb, textlines, verbose=verbose)
