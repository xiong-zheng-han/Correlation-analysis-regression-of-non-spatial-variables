"""Main entry point for the correlation analyzer application."""

import sys
from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main():
    """
    Main entry point for the application.

    Launches the PyQt6 GUI for the correlation analyzer.
    """
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("空间要素与鱼塘要素相关性分析工具")
    app.setOrganizationName("GeoAI Lab")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
