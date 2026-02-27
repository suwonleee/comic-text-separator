"""Main application window for comic-text-separator GUI."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import (
    QPropertyAnimation,
    QEasingCurve,
    Qt,
)
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.worker import PipelineController, detect_device
from gui.styles import apply_shadow

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
IMAGE_FILTER = "이미지 파일 (*.jpg *.jpeg *.png *.webp *.bmp *.tiff);;모든 파일 (*)"


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------

class CollapsibleSection(QWidget):
    """A collapsible section with animated expand/collapse."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._toggle_button = QToolButton()
        self._toggle_button.setObjectName("collapsible_toggle")
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_button.setText(title)
        self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.toggled.connect(self._on_toggled)

        self._content_area = QScrollArea()
        self._content_area.setFrameShape(QFrame.Shape.NoFrame)
        self._content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._content_area.setMaximumHeight(0)
        self._content_area.setMinimumHeight(0)
        self._content_area.setWidgetResizable(True)

        self._animation = QPropertyAnimation(self._content_area, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuart)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._content_area)

    def set_content_layout(self, content_layout: QVBoxLayout) -> None:
        """Set the inner content layout for this section."""
        widget = QWidget()
        widget.setLayout(content_layout)
        self._content_area.setWidget(widget)

    def _on_toggled(self, checked: bool) -> None:
        arrow = Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        self._toggle_button.setArrowType(arrow)

        content_widget = self._content_area.widget()
        target_height = content_widget.sizeHint().height() if checked else 0

        self._animation.stop()
        self._animation.setStartValue(self._content_area.maximumHeight())
        self._animation.setEndValue(target_height)
        self._animation.start()


# ---------------------------------------------------------------------------
# Drag-and-drop image list
# ---------------------------------------------------------------------------

class ImageDropList(QListWidget):
    """QListWidget subclass that accepts image file drops."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._overlay_visible = False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._overlay_visible = True
            self.setStyleSheet(
                "QListWidget#image_list {"
                "  border: 3px dashed #0d0d0d;"
                "  background-color: #f0f0f0;"
                "}"
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._overlay_visible = False
        self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent) -> None:
        self._overlay_visible = False
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        paths = []
        rejected = []
        for url in urls:
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                paths.append(path)
            elif ext:
                rejected.append(os.path.basename(path))
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
        if rejected:
            supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
            QMessageBox.warning(
                self.window(),
                "지원하지 않는 파일 형식",
                f"다음 파일은 지원하지 않는 형식입니다:\n"
                f"{', '.join(rejected)}\n\n"
                f"지원 형식: {supported}",
            )

    def _add_paths(self, paths: list[str]) -> None:
        """Add file paths, avoiding duplicates."""
        existing = {self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setToolTip(p)
                self.addItem(item)
                existing.add(p)

    def get_all_paths(self) -> list[str]:
        """Return all file paths in order."""
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() == 0 and not self._overlay_visible:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#888888"))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter,
                             "이미지를 드래그하여 추가하세요")
            painter.end()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("comic-text-separator")
        self.setMinimumSize(500, 600)
        self.resize(620, 780)

        self._controller = PipelineController(self)
        self._output_dir = os.path.abspath("output")

        self._init_ui()
        self._apply_shadows()
        self._set_idle_state()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # -- Header --
        header = QHBoxLayout()
        title_label = QLabel("comic-text-separator")
        title_label.setObjectName("title_label")
        header.addWidget(title_label)

        header.addStretch()

        device_id, device_name = detect_device()
        self._device_id = device_id
        self._device_badge = QLabel(f"🖥 {device_name}")
        self._device_badge.setObjectName("device_badge")
        header.addWidget(self._device_badge)
        root.addLayout(header)

        # -- Header separator --
        header_line = QFrame()
        header_line.setObjectName("header_line")
        header_line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(header_line)

        # -- Image list area --
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("이미지 추가")
        self._btn_add.setObjectName("btn_add")
        self._btn_add.clicked.connect(self._on_add_images)
        btn_row.addWidget(self._btn_add)

        self._btn_remove = QPushButton("선택 삭제")
        self._btn_remove.setObjectName("btn_remove")
        self._btn_remove.clicked.connect(self._on_remove_selected)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._image_list = ImageDropList()
        self._image_list.setObjectName("image_list")
        self._image_list.setMinimumHeight(150)
        root.addWidget(self._image_list, stretch=1)

        # -- Output options --
        fmt_layout = QHBoxLayout()
        fmt_label = QLabel("출력 형식:")
        fmt_layout.addWidget(fmt_label)

        self._fmt_group = QButtonGroup(self)
        self._radio_psd = QRadioButton("PSD")
        self._radio_json = QRadioButton("JSON")
        self._radio_both = QRadioButton("둘 다")
        self._radio_psd.setChecked(True)
        self._fmt_group.addButton(self._radio_psd, 0)
        self._fmt_group.addButton(self._radio_json, 1)
        self._fmt_group.addButton(self._radio_both, 2)
        fmt_layout.addWidget(self._radio_psd)
        fmt_layout.addWidget(self._radio_json)
        fmt_layout.addWidget(self._radio_both)
        fmt_layout.addStretch()
        root.addLayout(fmt_layout)

        # Output folder
        out_layout = QHBoxLayout()
        out_label = QLabel("출력 폴더:")
        out_layout.addWidget(out_label)

        self._output_edit = QLineEdit(self._output_dir)
        self._output_edit.setObjectName("output_edit")
        self._output_edit.setReadOnly(True)
        out_layout.addWidget(self._output_edit, stretch=1)

        self._btn_output = QPushButton("변경")
        self._btn_output.setObjectName("btn_output")
        self._btn_output.clicked.connect(self._on_change_output)
        out_layout.addWidget(self._btn_output)
        root.addLayout(out_layout)

        # -- Advanced options (collapsible) --
        self._advanced = CollapsibleSection("고급 옵션")
        adv_layout = QVBoxLayout()
        adv_layout.setContentsMargins(12, 10, 12, 10)
        adv_layout.setSpacing(8)

        # Detection size
        row = QHBoxLayout()
        row.addWidget(QLabel("Detection size:"))
        self._spin_det_size = QSpinBox()
        self._spin_det_size.setRange(512, 4096)
        self._spin_det_size.setSingleStep(256)
        self._spin_det_size.setValue(2048)
        row.addWidget(self._spin_det_size)
        row.addStretch()
        adv_layout.addLayout(row)

        # Text threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Text threshold:"))
        self._spin_text_thresh = QDoubleSpinBox()
        self._spin_text_thresh.setRange(0.1, 1.0)
        self._spin_text_thresh.setSingleStep(0.05)
        self._spin_text_thresh.setValue(0.5)
        row.addWidget(self._spin_text_thresh)
        row.addStretch()
        adv_layout.addLayout(row)

        # Box threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Box threshold:"))
        self._spin_box_thresh = QDoubleSpinBox()
        self._spin_box_thresh.setRange(0.1, 1.0)
        self._spin_box_thresh.setSingleStep(0.05)
        self._spin_box_thresh.setValue(0.7)
        row.addWidget(self._spin_box_thresh)
        row.addStretch()
        adv_layout.addLayout(row)

        # Inpainting
        row = QHBoxLayout()
        self._chk_inpaint = QCheckBox("인페인팅 (텍스트 제거)")
        self._chk_inpaint.setChecked(True)
        row.addWidget(self._chk_inpaint)
        row.addWidget(QLabel("크기:"))
        self._spin_inpaint_size = QSpinBox()
        self._spin_inpaint_size.setRange(512, 4096)
        self._spin_inpaint_size.setSingleStep(256)
        self._spin_inpaint_size.setValue(2048)
        row.addWidget(self._spin_inpaint_size)
        row.addStretch()
        adv_layout.addLayout(row)

        # Spacing correction
        self._chk_spacing = QCheckBox("띄어쓰기 교정")
        self._chk_spacing.setChecked(True)
        adv_layout.addWidget(self._chk_spacing)

        self._advanced.set_content_layout(adv_layout)
        root.addWidget(self._advanced)

        # -- Progress area --
        self._status_label = QLabel("")
        self._status_label.setObjectName("status_label")
        root.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("progress_bar")
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        root.addWidget(self._progress_bar)

        # -- Action buttons --
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton("처리 시작")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self._btn_start)

        self._btn_open_output = QPushButton("출력 폴더 열기")
        self._btn_open_output.setObjectName("btn_open_output")
        self._btn_open_output.clicked.connect(self._on_open_output)
        self._btn_open_output.setVisible(False)
        btn_layout.addWidget(self._btn_open_output)

        root.addLayout(btn_layout)

    # -----------------------------------------------------------------------
    # Neo-brutalism shadow effects
    # -----------------------------------------------------------------------

    def _apply_shadows(self) -> None:
        """Apply solid drop-shadow effects to key interactive widgets."""
        apply_shadow(self._btn_start, 2, 2)
        apply_shadow(self._btn_add, 2, 2)
        apply_shadow(self._btn_remove, 2, 2)
        apply_shadow(self._image_list, 2, 2)
        apply_shadow(self._btn_open_output, 2, 2)
        apply_shadow(self._progress_bar, 2, 2)

    # -----------------------------------------------------------------------
    # Config assembly
    # -----------------------------------------------------------------------

    def _build_config(self) -> dict:
        fmt_map = {0: 'psd', 1: 'json', 2: 'both'}
        return {
            'output_dir': self._output_dir,
            'format': fmt_map.get(self._fmt_group.checkedId(), 'psd'),
            'detection_size': self._spin_det_size.value(),
            'text_threshold': self._spin_text_thresh.value(),
            'box_threshold': self._spin_box_thresh.value(),
            'inpaint': self._chk_inpaint.isChecked(),
            'inpainting_size': self._spin_inpaint_size.value(),
            'correct_spacing': self._chk_spacing.isChecked(),
            'device': self._device_id,
        }

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def _set_idle_state(self) -> None:
        self._set_inputs_enabled(True)
        self._btn_start.setText("처리 시작")
        self._btn_start.clicked.disconnect()
        self._btn_start.clicked.connect(self._on_start)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._status_label.setText("")

    def _set_processing_state(self, total: int) -> None:
        self._set_inputs_enabled(False)
        self._btn_start.setText("취소")
        self._btn_start.setEnabled(True)
        self._btn_start.clicked.disconnect()
        self._btn_start.clicked.connect(self._on_cancel)
        self._btn_open_output.setVisible(False)
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(0)

    def _set_done_state(self, done: int, total: int, errors: int) -> None:
        self._set_inputs_enabled(True)
        self._btn_start.setText("처리 시작")
        self._btn_start.clicked.disconnect()
        self._btn_start.clicked.connect(self._on_start)

        error_msg = f" (오류 {errors}건)" if errors else ""
        self._status_label.setText(f"완료! {done}/{total} 파일 처리됨{error_msg}")
        self._btn_open_output.setVisible(True)

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self._btn_add.setEnabled(enabled)
        self._btn_remove.setEnabled(enabled)
        self._image_list.setEnabled(enabled)
        self._radio_psd.setEnabled(enabled)
        self._radio_json.setEnabled(enabled)
        self._radio_both.setEnabled(enabled)
        self._btn_output.setEnabled(enabled)
        self._spin_det_size.setEnabled(enabled)
        self._spin_text_thresh.setEnabled(enabled)
        self._spin_box_thresh.setEnabled(enabled)
        self._chk_inpaint.setEnabled(enabled)
        self._spin_inpaint_size.setEnabled(enabled)
        self._chk_spacing.setEnabled(enabled)

    # -----------------------------------------------------------------------
    # Slots — UI actions
    # -----------------------------------------------------------------------

    def _on_add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "이미지 선택", "", IMAGE_FILTER)
        if paths:
            valid = []
            rejected = []
            for p in paths:
                ext = os.path.splitext(p)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    valid.append(p)
                else:
                    rejected.append(os.path.basename(p))
            if valid:
                self._image_list._add_paths(valid)
            if rejected:
                supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
                QMessageBox.warning(
                    self,
                    "지원하지 않는 파일 형식",
                    f"다음 파일은 지원하지 않는 형식입니다:\n"
                    f"{', '.join(rejected)}\n\n"
                    f"지원 형식: {supported}",
                )

    def _on_remove_selected(self) -> None:
        for item in reversed(self._image_list.selectedItems()):
            row = self._image_list.row(item)
            self._image_list.takeItem(row)

    def _on_change_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택", self._output_dir)
        if folder:
            self._output_dir = folder
            self._output_edit.setText(folder)

    def _on_start(self) -> None:
        files = self._image_list.get_all_paths()
        if not files:
            QMessageBox.warning(self, "알림", "처리할 이미지를 추가해주세요.")
            return

        self._error_count = 0
        self._done_count = 0
        total = len(files)
        config = self._build_config()

        self._set_processing_state(total)

        worker = self._controller.start(files, config)

        # Connect worker signals
        worker.model_loading.connect(self._on_model_loading)
        worker.model_ready.connect(lambda: self._on_model_ready(total))
        worker.file_started.connect(self._on_file_started)
        worker.file_done.connect(self._on_file_done)
        worker.file_error.connect(self._on_file_error)
        worker.overall_progress.connect(self._on_overall_progress)
        worker.finished.connect(lambda: self._on_finished(total))
        worker.error.connect(self._on_critical_error)

    def _on_cancel(self) -> None:
        self._controller.cancel()
        self._status_label.setText("취소 중...")
        self._btn_start.setEnabled(False)

    def _on_open_output(self) -> None:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', self._output_dir])
        elif sys.platform == 'win32':
            subprocess.Popen(['explorer', self._output_dir])
        else:
            subprocess.Popen(['xdg-open', self._output_dir])

    # -----------------------------------------------------------------------
    # Slots — worker signals
    # -----------------------------------------------------------------------

    def _on_model_loading(self) -> None:
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._status_label.setText("모델 로딩 중...")

    def _on_model_ready(self, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(0)
        self._status_label.setText("모델 준비 완료")

    def _on_file_started(self, idx: int, path: str) -> None:
        name = os.path.basename(path)
        self._status_label.setText(f"처리 중: {name}")

    def _on_file_done(self, idx: int, path: str) -> None:
        self._done_count += 1

    def _on_file_error(self, idx: int, msg: str) -> None:
        self._error_count += 1
        self._status_label.setText(f"오류: {msg}")

    def _on_overall_progress(self, done: int, total: int) -> None:
        self._progress_bar.setValue(done)

    def _on_finished(self, total: int) -> None:
        self._set_done_state(self._done_count, total, self._error_count)

    def _on_critical_error(self, msg: str) -> None:
        QMessageBox.critical(self, "오류", f"처리 중 심각한 오류가 발생했습니다:\n{msg}")
