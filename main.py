#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re

from PIL import Image

from adapters.detector import MangaTranslatorDetector
from adapters.ocr_engine import MangaTranslatorOcr
from adapters.textline_merger import MangaTranslatorMerger
from adapters.textblock_mapper import TextBlockMapper
from adapters.inpainter import MangaTranslatorInpainter
from adapters.kss_spacing import KssSpacingCorrector
from infrastructure.json_exporter import save_json
from infrastructure.jsx_exporter import save_jsx
from infrastructure.psd_exporter import save_psd
from use_cases.extract_text import ExtractTextUseCase


def _natural_sort_key(s: str):
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", s)
    ]


def collect_images(path: str) -> list[str]:
    SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if os.path.isfile(path):
        return [path]
    images = []
    for root, _, files in os.walk(path):
        for f in sorted(files, key=_natural_sort_key):
            if os.path.splitext(f)[1].lower() in SUPPORTED:
                images.append(os.path.join(root, f))
    return images


async def run(args: argparse.Namespace) -> None:
    device = "cpu"
    if args.use_gpu:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda"

    detector = MangaTranslatorDetector(
        detection_size=args.detection_size,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        device=device,
    )
    ocr = MangaTranslatorOcr(device=device)
    merger = MangaTranslatorMerger()
    mapper = TextBlockMapper()

    inpainter = None
    if args.inpaint:
        inpainter = MangaTranslatorInpainter(
            inpainting_size=args.inpainting_size,
            device=device,
        )

    spacing_corrector = KssSpacingCorrector() if args.correct_spacing else None

    use_case = ExtractTextUseCase(
        detector=detector,
        ocr=ocr,
        merger=merger,
        mapper=mapper,
        inpainter=inpainter,
        spacing_corrector=spacing_corrector,
    )

    images = collect_images(args.input)
    if not images:
        logging.error(f"No images found at: {args.input}")
        return

    logging.info(f"Found {len(images)} image(s)")

    await use_case.prepare()

    for img_path in images:
        try:
            result, inpainted_img = await use_case.execute(
                img_path, verbose=args.verbose
            )

            if inpainted_img is not None:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                inpainted_path = os.path.join(args.output, f"{base_name}_inpainted.png")
                os.makedirs(args.output, exist_ok=True)
                Image.fromarray(inpainted_img).save(inpainted_path)
                result.inpainted_image_path = os.path.abspath(inpainted_path)
                logging.info(f"  Saved inpainted: {inpainted_path}")

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            saved: list[str] = []

            if args.format in ("json", "both"):
                json_path = os.path.join(args.output, f"{base_name}.json")
                save_json(result, json_path)
                logging.info(f"  Saved JSON: {json_path} ({len(result.regions)} regions)")
                saved.append(json_path)

            if args.format in ("psd", "both"):
                psd_path = os.path.join(args.output, f"{base_name}.psd")
                save_psd(result, psd_path, inpainted_img=inpainted_img)
                saved.append(psd_path)

                jsx_path = os.path.join(args.output, f"{base_name}.jsx")
                if save_jsx(result, jsx_path):
                    saved.append(jsx_path)

            for path in saved:
                print(f"✓ {os.path.basename(img_path)} → {path}")

        except Exception as e:
            logging.error(f"✗ {os.path.basename(img_path)}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\nDone. Results in: {os.path.abspath(args.output)}/")
    if args.format in ("psd", "both"):
        print("Open PSD files in Photopea (https://www.photopea.com) for editing.")
        print("JSX script → Photopea에서 File > Script > 붙여넣기 > Run → 편집 가능한 텍스트 레이어 생성")


def main():
    parser = argparse.ArgumentParser(
        prog="comic-text-separator",
        description="Extract and separate text layers from comic/conti images to PSD or JSON",
    )
    parser.add_argument(
        "-i", "--input", default="input",
        help="Path to image file or folder (default: input/)",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "-f", "--format", default="psd",
        choices=["psd", "json", "both"],
        help="Output format (default: psd)",
    )
    parser.add_argument(
        "--detection-size", type=int, default=2048,
        help="Detection image size (default: 2048)",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.5,
        help="Text detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.7,
        help="Bounding box threshold (default: 0.7)",
    )
    parser.add_argument(
        "--inpaint", action="store_true",
        help="Remove detected text from image (adds inpainted layer to PSD)",
    )
    parser.add_argument(
        "--inpainting-size", type=int, default=2048,
        help="Inpainting image size (default: 2048)",
    )
    parser.add_argument(
        "--correct-spacing", action="store_true",
        help="Correct Korean word spacing in OCR output (saves to JSON text_corrected field)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Use GPU (auto-detect CUDA or MPS)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
