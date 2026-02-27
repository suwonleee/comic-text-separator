"""Neo-Brutalism design system for comic-text-separator GUI.

Color palette, global QSS stylesheet, and shadow helper.
"""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BASE_BLACK = "#0d0d0d"
BASE_WHITE = "#fdfdfd"

GRAY_LIGHT = "#f0f0f0"
GRAY_MID = "#e0e0e0"
GRAY_TEXT = "#888888"

# ---------------------------------------------------------------------------
# Shadow helper
# ---------------------------------------------------------------------------


def apply_shadow(
    widget: QWidget,
    offset_x: float = 2,
    offset_y: float = 2,
    blur: float = 1,
    color: str = "#c0c0c0",
) -> QGraphicsDropShadowEffect:
    """Apply a neo-brutalist solid shadow effect to *widget*.

    Each widget can only hold one QGraphicsEffect, so calling this
    replaces any previous effect on the widget.
    """
    effect = QGraphicsDropShadowEffect(widget)
    effect.setOffset(offset_x, offset_y)
    effect.setBlurRadius(blur)
    effect.setColor(QColor(color))
    widget.setGraphicsEffect(effect)
    return effect


# ---------------------------------------------------------------------------
# Global QSS stylesheet
# ---------------------------------------------------------------------------

GLOBAL_STYLESHEET = f"""
/* ===== Base ===== */

QMainWindow {{
    background-color: {BASE_WHITE};
}}

QWidget {{
    font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
}}

QToolTip {{
    background-color: {BASE_BLACK};
    color: {BASE_WHITE};
    border: 2px solid {BASE_BLACK};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ===== QPushButton (general) ===== */

QPushButton {{
    background-color: {BASE_WHITE};
    color: {BASE_BLACK};
    border: 2px solid {BASE_BLACK};
    border-radius: 8px;
    font-weight: bold;
    font-size: 13px;
    padding: 8px 16px;
    min-height: 30px;
}}

QPushButton:hover {{
    background-color: {GRAY_LIGHT};
}}

QPushButton:pressed {{
    background-color: {GRAY_MID};
}}

QPushButton:disabled {{
    background-color: {GRAY_LIGHT};
    color: {GRAY_TEXT};
    border-color: {GRAY_TEXT};
}}

/* ===== Primary CTA — 처리 시작 ===== */

QPushButton#btn_start {{
    background-color: {BASE_BLACK};
    color: {BASE_WHITE};
    border: 2px solid {BASE_BLACK};
    border-radius: 8px;
    font-size: 14pt;
    font-weight: bold;
    min-height: 44px;
    padding: 8px 24px;
}}

QPushButton#btn_start:hover {{
    background-color: #333333;
}}

QPushButton#btn_start:pressed {{
    background-color: #555555;
}}

QPushButton#btn_start:disabled {{
    background-color: {GRAY_TEXT};
    color: {GRAY_MID};
    border-color: {GRAY_TEXT};
}}

/* ===== 출력 폴더 열기 ===== */

QPushButton#btn_open_output {{
    min-height: 44px;
    font-size: 13px;
    padding: 8px 20px;
}}

/* ===== QListWidget (image list) ===== */

QListWidget#image_list {{
    background-color: {BASE_WHITE};
    border: 2px solid {BASE_BLACK};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}

QListWidget#image_list::item {{
    padding: 8px 10px;
    border-radius: 4px;
    color: {BASE_BLACK};
}}

QListWidget#image_list::item:selected {{
    background-color: {GRAY_LIGHT};
    color: {BASE_BLACK};
}}

QListWidget#image_list::item:hover:!selected {{
    background-color: {GRAY_LIGHT};
}}

/* ===== QLineEdit ===== */

QLineEdit {{
    background-color: {BASE_WHITE};
    color: {BASE_BLACK};
    border: 2px solid {BASE_BLACK};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}}

QLineEdit:read-only {{
    background-color: {GRAY_LIGHT};
}}

QLineEdit:focus {{
    border-color: {BASE_BLACK};
}}

/* ===== QRadioButton ===== */

QRadioButton {{
    font-size: 13px;
    font-weight: bold;
    spacing: 8px;
    color: {BASE_BLACK};
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {BASE_BLACK};
    border-radius: 10px;
    background-color: {BASE_WHITE};
}}

QRadioButton::indicator:checked {{
    background-color: {BASE_BLACK};
    border: 2px solid {BASE_BLACK};
}}

QRadioButton::indicator:hover {{
    border-color: {BASE_BLACK};
    background-color: {GRAY_LIGHT};
}}

QRadioButton::indicator:checked:hover {{
    background-color: {BASE_BLACK};
}}

/* ===== QCheckBox ===== */

QCheckBox {{
    font-size: 13px;
    spacing: 8px;
    color: {BASE_BLACK};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {BASE_BLACK};
    border-radius: 4px;
    background-color: {BASE_WHITE};
}}

QCheckBox::indicator:checked {{
    background-color: {BASE_BLACK};
    border: 2px solid {BASE_BLACK};
}}

QCheckBox::indicator:hover {{
    background-color: {GRAY_LIGHT};
}}

QCheckBox::indicator:checked:hover {{
    background-color: {BASE_BLACK};
}}

/* ===== QSpinBox / QDoubleSpinBox ===== */

QSpinBox, QDoubleSpinBox {{
    background-color: {BASE_WHITE};
    color: {BASE_BLACK};
    border: 2px solid {BASE_BLACK};
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 13px;
    min-height: 28px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 2px solid {BASE_BLACK};
    border-bottom: 1px solid {GRAY_MID};
    border-top-right-radius: 4px;
    background-color: {GRAY_LIGHT};
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background-color: {GRAY_MID};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 2px solid {BASE_BLACK};
    border-top: 1px solid {GRAY_MID};
    border-bottom-right-radius: 4px;
    background-color: {GRAY_LIGHT};
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {GRAY_MID};
}}

/* ===== QProgressBar ===== */

QProgressBar {{
    background-color: {GRAY_MID};
    border: 2px solid {BASE_BLACK};
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    font-size: 12px;
    min-height: 26px;
    color: {BASE_BLACK};
}}

QProgressBar::chunk {{
    background-color: {BASE_BLACK};
    border-radius: 6px;
}}

/* ===== Collapsible toggle ===== */

QToolButton#collapsible_toggle {{
    border: 2px solid {BASE_BLACK};
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: bold;
    font-size: 13px;
    background-color: {GRAY_LIGHT};
    color: {BASE_BLACK};
    text-align: left;
}}

QToolButton#collapsible_toggle:hover {{
    background-color: {GRAY_MID};
}}

QToolButton#collapsible_toggle:checked {{
    background-color: {GRAY_MID};
    border-bottom-left-radius: 0px;
    border-bottom-right-radius: 0px;
}}

/* ===== Labels ===== */

QLabel#title_label {{
    font-size: 18pt;
    font-weight: bold;
    color: {BASE_BLACK};
}}

QLabel#device_badge {{
    background-color: {GRAY_LIGHT};
    border: 2px solid {BASE_BLACK};
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: bold;
    font-size: 12px;
    color: {BASE_BLACK};
}}

QLabel#status_label {{
    font-size: 13px;
    font-weight: bold;
    color: {BASE_BLACK};
    padding: 2px 0px;
}}

QLabel {{
    color: {BASE_BLACK};
}}

/* ===== QScrollBar (neo-brutalist thick handle) ===== */

QScrollBar:vertical {{
    border: none;
    background-color: {GRAY_LIGHT};
    width: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {BASE_BLACK};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: #333333;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    border: none;
    background-color: {GRAY_LIGHT};
    height: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {BASE_BLACK};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: #333333;
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* ===== QScrollArea (collapsible content) ===== */

QScrollArea {{
    background: transparent;
    border: none;
}}

/* ===== QFrame separator ===== */

QFrame#header_line {{
    background-color: {BASE_BLACK};
    min-height: 2px;
    max-height: 2px;
}}
"""
