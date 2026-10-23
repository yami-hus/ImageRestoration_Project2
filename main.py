import sys
from PySide6 import QtWidgets
from gui.style import apply_theme
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1200, 780)
    win.show()
    sys.exit(app.exec())
