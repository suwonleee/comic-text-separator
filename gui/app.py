"""GUI entry point for comic-text-separator."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.styles import GLOBAL_STYLESHEET


def gui_main() -> None:
    """Launch the desktop GUI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui_main()
