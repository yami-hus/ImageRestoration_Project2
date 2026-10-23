"""Histogram kênh H (tô đúng màu hue) + vùng ngưỡng H đang chọn."""
import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore


def compute_h_hist(img_bgr):
    """Histogram H (180 bin), bỏ qua điểm quá xám/tối để đồ thị có ý nghĩa."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 30, 30], np.uint8), np.array([179, 255, 255], np.uint8))
    return cv2.calcHist([hsv], [0], m, [180], [0, 180]).flatten()


class HueHistogram(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(110)
        self.hist = None
        self.hrange = (0, 179)

    def set_data(self, hist):
        self.hist = hist
        self.update()

    def set_range(self, hmin, hmax):
        self.hrange = (hmin, hmax)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor("#18181b"))
        bw = w / 180.0
        if self.hist is not None and self.hist.max() > 0:
            mx = float(self.hist.max())
            for i, v in enumerate(self.hist):
                bh = int(v / mx * (h - 4))
                p.fillRect(QtCore.QRectF(i * bw, h - bh, bw + 1, bh),
                           QtGui.QColor.fromHsv(i * 2, 255, 255))
        lo, hi = self.hrange
        spans = [(lo, hi)] if lo <= hi else [(lo, 179), (0, hi)]
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, 70))
        for a, b in spans:
            p.drawRect(QtCore.QRectF(a * bw, 0, (b - a + 1) * bw, h))
