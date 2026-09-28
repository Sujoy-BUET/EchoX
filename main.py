"""EchoX: an audio signal workbench.

Run:  python main.py [optional audio file]
"""
import sys

from PySide6.QtWidgets import QApplication

from echox import theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EchoX")
    theme.apply(app)
    theme.configure_pyqtgraph()

    from echox.ui.window import MainWindow   # after pyqtgraph is configured

    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
