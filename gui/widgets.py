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


class NoWheelComboBox(_NoWheelMixin, QtWidgets.QComboBox):
    def __init__(self, *a):
        super().__init__(*a)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)


class WheelGuard(QtCore.QObject):
 

    def __init__(self, get_scroll_area):
        super().__init__()
        self._get_scroll_area = get_scroll_area

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            area = self._get_scroll_area()
            if area is not None:
                QtWidgets.QApplication.sendEvent(area.verticalScrollBar(), ev)
            return True
        return False
