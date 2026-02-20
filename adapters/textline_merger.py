from __future__ import annotations

from engine.merger import merge_to_blocks


class MangaTranslatorMerger:

    async def merge(
        self, textlines: list, width: int, height: int, verbose: bool = False
    ) -> list:
        return await merge_to_blocks(textlines, width, height, verbose=verbose)
