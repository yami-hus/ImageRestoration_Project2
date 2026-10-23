"""Giao diện chung: bảng màu trung tính (không ám màu lên ảnh) + QSS.
Điểm nhấn duy nhất: các thanh trượt H/S/V tô đúng dải màu mà nó điều khiển."""
from PySide6 import QtGui, QtWidgets

BG = "#232326"        # nền cửa sổ
PANEL = "#2d2d32"     # nền thẻ
BORDER = "#3b3b42"
CANVAS = "#18181b"    # nền khung ảnh
TEXT = "#e8e8ec"
MUTED = "#9a9aa6"
ACCENT = "#7b8cde"
ACCENT_HOVER = "#93a2ea"
ON_ACCENT = "#14142a"
DANGER = "#e58a8a"

QSS = f"""
QMainWindow, QWidget#root {{ background: {BG}; }}
QScrollArea {{ background: {BG}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #55555f; border-radius: 5px; min-height: 32px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #6a6a76; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #55555f; border-radius: 5px; min-width: 32px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: #6a6a76; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
QLabel {{ background: transparent; }}

QLabel#appTitle {{ font-size: 17pt; font-weight: 700; }}
QLabel#subtitle {{ color: {MUTED}; }}
QLabel#source {{ color: {MUTED}; }}
QLabel#cardTitle {{ font-weight: 600; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#rowName {{ color: {MUTED}; min-width: 44px; }}

QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#imageView {{ background: {CANVAS}; color: {MUTED}; border-radius: 6px; }}

QPushButton {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 8px 16px; color: {TEXT}; min-height: 20px;
}}
QPushButton:hover {{ background: #38383f; border-color: #50505a; }}
QPushButton:pressed {{ background: #26262a; }}
QPushButton[variant="primary"] {{
    background: {ACCENT}; border-color: {ACCENT}; color: {ON_ACCENT}; font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[variant="danger"] {{ color: {DANGER}; }}
QPushButton[variant="danger"]:hover {{ background: #3d2c30; border-color: #7a4850; }}

QComboBox, QSpinBox {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 10px; color: {TEXT}; min-height: 20px;
}}
QComboBox:hover, QSpinBox:hover {{ border-color: #50505a; }}
QComboBox QAbstractItemView {{
    background: {PANEL}; border: 1px solid {BORDER}; outline: 0;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT};
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #55555f; background: {CANVAS};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {BORDER}; }}
QSlider[kind="h"]::groove:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff0000, stop:0.17 #ffff00, stop:0.33 #00ff00,
        stop:0.5 #00ffff, stop:0.67 #0000ff, stop:0.83 #ff00ff, stop:1 #ff0000);
}}
QSlider[kind="s"]::groove:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d8d8d8, stop:1 {ACCENT});
}}
QSlider[kind="v"]::groove:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #000000, stop:1 #ffffff);
}}
QSlider::handle:horizontal {{
    background: #ffffff; border: 2px solid #2d2d32; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ border-color: {ACCENT}; }}

QLabel#valuePill {{
    background: {CANVAS}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 2px 6px; min-width: 28px;
}}
QLabel#metricValue {{ font-size: 18pt; font-weight: 700; min-height: 36px; }}
QLabel#metricCaption {{ color: {MUTED}; }}

QStatusBar {{ background: {BG}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
"""


def apply_theme(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    pal = QtGui.QPalette()
    R = QtGui.QPalette.ColorRole
    for role, color in (
        (R.Window, BG), (R.WindowText, TEXT), (R.Base, CANVAS), (R.AlternateBase, PANEL),
        (R.Text, TEXT), (R.Button, PANEL), (R.ButtonText, TEXT),
        (R.ToolTipBase, PANEL), (R.ToolTipText, TEXT),
        (R.Highlight, ACCENT), (R.HighlightedText, ON_ACCENT),
    ):
        pal.setColor(role, QtGui.QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
