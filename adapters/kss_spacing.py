from __future__ import annotations

import logging
import re

from domain.models import ExtractionResult

logger = logging.getLogger(__name__)

_kss_loaded = False

_HANGUL_RE = re.compile(r'[\uAC00-\uD7AF\u3131-\u3163]')


def _ensure_kss():
    global _kss_loaded
    if not _kss_loaded:
        import kss  # noqa: F401
        _kss_loaded = True


def _is_korean(text: str, source_lang: str) -> bool:
    if source_lang == "ko":
        return True
    if source_lang:
        return False
    return bool(_HANGUL_RE.search(text))


class KssSpacingCorrector:

    def correct(self, result: ExtractionResult) -> ExtractionResult:
        _ensure_kss()
        import kss

        corrected_count = 0
        skipped_count = 0
        for region in result.regions:
            text = region.text
            if not text or not text.strip():
                region.text_corrected = text
                continue

            if not _is_korean(text, region.source_lang):
                region.text_corrected = text
                skipped_count += 1
                continue

            corrected = kss.correct_spacing(text)
            if corrected != text:
                corrected_count += 1
            region.text_corrected = corrected

        total = len(result.regions)
        logger.info(f"  Spacing corrected: {corrected_count}/{total} regions modified, {skipped_count} non-Korean skipped")
        return result
