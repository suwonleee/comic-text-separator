"""Background processing worker for the GUI pipeline.

Uses QObject + moveToThread() pattern for non-blocking UI.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from PIL import Image

logger = logging.getLogger(__name__)


def detect_device() -> tuple[str, str]:
    """Returns (device_id, display_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return 'cuda', f'NVIDIA GPU ({name})'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'Apple Silicon (MPS)'
    except Exception:
        pass
    return 'cpu', 'CPU'


class ProcessingWorker(QObject):
    """Runs the ML pipeline in a background thread."""

    # Signals
    file_started = Signal(int, str)
    file_done = Signal(int, str)
    file_error = Signal(int, str)
    overall_progress = Signal(int, int)
    model_loading = Signal()
    model_ready = Signal()
    finished = Signal()
    error = Signal(str)

    def __init__(self, files: list[str], config: dict) -> None:
        super().__init__()
        self._files = files
        self._config = config
        self._abort = False

    def cancel(self) -> None:
        """Thread-safe abort flag setter."""
        self._abort = True

    @Slot()
    def run(self) -> None:
        """Main processing entry point — runs in worker thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            loop.close()
            self.finished.emit()

    async def _run_async(self) -> None:
        """Async pipeline execution mirroring main.py logic."""
        cfg = self._config
        device = cfg.get('device', 'cpu')
        output_dir = cfg.get('output_dir', 'output')
        fmt = cfg.get('format', 'psd')

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

        # Build adapters identical to main.py
        detector = MangaTranslatorDetector(
            detection_size=cfg.get('detection_size', 2048),
            text_threshold=cfg.get('text_threshold', 0.5),
            box_threshold=cfg.get('box_threshold', 0.7),
            device=device,
        )
        ocr = MangaTranslatorOcr(device=device)
        merger = MangaTranslatorMerger()
        mapper = TextBlockMapper()

        inpainter = None
        if cfg.get('inpaint', True):
            inpainter = MangaTranslatorInpainter(
                inpainting_size=cfg.get('inpainting_size', 2048),
                device=device,
            )

        spacing_corrector = KssSpacingCorrector() if cfg.get('correct_spacing', True) else None

        use_case = ExtractTextUseCase(
            detector=detector,
            ocr=ocr,
            merger=merger,
            mapper=mapper,
            inpainter=inpainter,
            spacing_corrector=spacing_corrector,
        )

        # Load models
        self.model_loading.emit()
        await use_case.prepare()
        self.model_ready.emit()

        if self._abort:
            return

        total = len(self._files)
        done_count = 0

        for idx, img_path in enumerate(self._files):
            if self._abort:
                return

            self.file_started.emit(idx, img_path)

            try:
                result, inpainted_img = await use_case.execute(img_path, verbose=False)

                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                # Save inpainted PNG
                if inpainted_img is not None:
                    inpainted_path = os.path.join(output_dir, f"{base_name}_inpainted.png")
                    Image.fromarray(inpainted_img).save(inpainted_path)
                    result.inpainted_image_path = os.path.abspath(inpainted_path)

                # Save JSON
                if fmt in ('json', 'both'):
                    json_path = os.path.join(output_dir, f"{base_name}.json")
                    save_json(result, json_path)

                # Save PSD + JSX
                if fmt in ('psd', 'both'):
                    psd_path = os.path.join(output_dir, f"{base_name}.psd")
                    save_psd(result, psd_path, inpainted_img=inpainted_img)
                    jsx_path = os.path.join(output_dir, f"{base_name}.jsx")
                    save_jsx(result, jsx_path)

                done_count += 1
                self.file_done.emit(idx, img_path)

            except Exception as exc:
                self.file_error.emit(idx, f"{os.path.basename(img_path)}: {exc}")

            self.overall_progress.emit(idx + 1, total)


class PipelineController(QObject):
    """Manages the worker thread lifecycle."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ProcessingWorker] = None

    @property
    def worker(self) -> Optional[ProcessingWorker]:
        return self._worker

    def start(self, files: list[str], config: dict) -> ProcessingWorker:
        """Create thread + worker, wire signals, and start processing."""
        self._thread = QThread()
        self._worker = ProcessingWorker(files, config)
        self._worker.moveToThread(self._thread)

        # Wire thread start to worker run
        self._thread.started.connect(self._worker.run)

        # Cleanup on finish
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()
        return self._worker

    def cancel(self) -> None:
        """Set abort flag on worker."""
        if self._worker is not None:
            self._worker.cancel()

    def _on_thread_finished(self) -> None:
        """Clean up references after thread ends."""
        self._thread = None
        self._worker = None
