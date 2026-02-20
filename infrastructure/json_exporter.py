from __future__ import annotations

import os

from domain.models import ExtractionResult


def save_json(result: ExtractionResult, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.to_json(indent=2))
    return output_path
