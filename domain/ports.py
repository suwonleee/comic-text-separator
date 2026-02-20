"""
Port interfaces (driven/secondary ports) for the text extraction domain.

All adapters must conform to these Protocols.
The use-case layer depends ONLY on these abstractions.
"""
from __future__ import annotations

from typing import Protocol, Optional, Tuple, List, runtime_checkable

import numpy as np

from .models import ExtractionResult


@runtime_checkable
class DetectorPort(Protocol):
    """Detect text regions in an image."""

    async def prepare(self) -> None: ...

    async def detect(
        self, img_rgb: np.ndarray, verbose: bool = False
    ) -> Tuple[list, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns
        -------
        textlines : list       – detected textline objects
        mask_raw  : ndarray|None – raw detection mask (needed for inpainting)
        mask      : ndarray|None – processed mask
        """
        ...


@runtime_checkable
class OcrPort(Protocol):
    """Run OCR on detected textlines."""

    async def prepare(self) -> None: ...

    async def recognize(
        self, img_rgb: np.ndarray, textlines: list, verbose: bool = False
    ) -> list:
        """Return textlines with ``.text`` populated."""
        ...


@runtime_checkable
class TextlineMergerPort(Protocol):
    """Merge individual textlines into logical text blocks."""

    async def merge(
        self, textlines: list, width: int, height: int, verbose: bool = False
    ) -> list:
        """Return merged text-block objects."""
        ...


@runtime_checkable
class TextBlockMapperPort(Protocol):
    """Map external text-block objects to our clean ExtractionResult model."""

    def map(
        self,
        image_path: str,
        img_width: int,
        img_height: int,
        text_regions: list,
    ) -> ExtractionResult: ...


@runtime_checkable
class InpainterPort(Protocol):
    """Remove detected text from images via inpainting."""

    async def prepare(self) -> None: ...

    async def refine_mask(
        self,
        text_regions: list,
        img_rgb: np.ndarray,
        mask_raw: np.ndarray,
    ) -> np.ndarray: ...

    async def inpaint(
        self, img_rgb: np.ndarray, mask: np.ndarray
    ) -> np.ndarray: ...


@runtime_checkable
class SpacingCorrectorPort(Protocol):
    """Post-process OCR text (e.g. Korean spacing correction)."""

    def correct(self, result: ExtractionResult) -> ExtractionResult: ...
