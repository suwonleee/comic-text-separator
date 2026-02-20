"""Generate Photopea/Photoshop ExtendScript (.jsx) for editable text layers."""

from __future__ import annotations

import logging
import os

from domain.models import ExtractionResult, TextRegionData

logger = logging.getLogger(__name__)


def _escape_js_double_quote(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\r")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _build_text_layer_jsx(idx: int, region: TextRegionData) -> str:
    text = region.text_corrected or region.text
    if not text.strip():
        return ""

    escaped = _escape_js_double_quote(text)
    r, g, b = region.fg_color
    fs = max(region.font_size, 10)
    w = max(region.width, 20)
    h = max(region.height, 20)

    preview = text[:20].replace("\n", " ")
    name = _escape_js_double_quote(f"Text {idx}: {preview}")

    direction = ""
    if region.direction == "v":
        direction = "\n  t.direction = Direction.VERTICAL;"

    # Position fallback for POINTTEXT: shift y down by ~80% of font size (baseline)
    fallback_y = region.y + int(fs * 0.8)

    return f"""(function() {{
  var layer = textGroup.artLayers.add();
  layer.kind = LayerKind.TEXT;
  var t = layer.textItem;{direction}
  t.size = new UnitValue({fs}, "px");
  try {{
    t.kind = TextType.PARAGRAPHTEXT;
    t.position = [new UnitValue({region.x}, "px"), new UnitValue({region.y}, "px")];
    t.width = new UnitValue({w}, "px");
    t.height = new UnitValue({h}, "px");
  }} catch(e) {{
    t.position = [new UnitValue({region.x}, "px"), new UnitValue({fallback_y}, "px")];
  }}
  t.contents = "{escaped}";
  try {{ t.font = FONT_NAME; }} catch(e) {{}}
  var c = new SolidColor();
  c.rgb.red = {r};
  c.rgb.green = {g};
  c.rgb.blue = {b};
  t.color = c;
  layer.name = "{name}";
}})();
"""


def save_jsx(result: ExtractionResult, output_path: str) -> str:
    """Returns output path, or empty string if no text regions exist."""
    if not result.regions:
        logger.info("No text regions — JSX not generated")
        return ""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    blocks: list[str] = []
    for i, region in enumerate(result.regions):
        block = _build_text_layer_jsx(i + 1, region)
        if block:
            blocks.append(block)

    if not blocks:
        return ""

    header = f"""\
// comic-text-separator — Editable Text Layer Script
// --------------------------------------------------
// Photopea: PSD를 연 상태에서 File > Script > 이 내용 붙여넣기 > Run
//
// 텍스트 {len(blocks)}개 | 폰트 변경: 아래 FONT_NAME 수정

var FONT_NAME = "NanumGothic";

var doc = app.activeDocument;

// 기존 래스터 텍스트 그룹 삭제
for (var i = doc.layerSets.length - 1; i >= 0; i--) {{
  if (doc.layerSets[i].name === "Text Layers") {{
    doc.layerSets[i].remove();
  }}
}}

// 편집 가능한 텍스트 레이어 그룹 생성
var textGroup = doc.layerSets.add();
textGroup.name = "Text Layers (Editable)";
"""

    footer = f'\nalert("{len(blocks)}개 편집 가능한 텍스트 레이어 생성 완료");\n'

    content = header + "\n" + "\n".join(blocks) + footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"  Saved JSX: {output_path} ({len(blocks)} text layers)")
    return output_path
