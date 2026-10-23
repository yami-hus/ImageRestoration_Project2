"""Ô nhập không bị đổi giá trị khi người dùng chỉ đang lăn chuột để cuộn trang.
Chỉ nhận lăn chuột khi ô đó đang được chọn (có focus)."""
from PySide6 import QtCore, QtWidgets


class _NoWheelMixin:
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()  # chuyển cho vùng cuộn bên ngoài


class NoWheelSlider(_NoWheelMixin, QtWidgets.QSlider):
    def __init__(self, *a):
        super().__init__(*a)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)


class NoWheelSpinBox(_NoWheelMixin, QtWidgets.QSpinBox):
    def __init__(self, *a):
        super().__init__(*a)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)


class NoWheelComboBox(_NoWheelMixin, QtWidgets.QComboBox):
    def __init__(self, *a):
        super().__init__(*a)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
