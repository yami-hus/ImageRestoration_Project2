import cv2
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore


def to_qpixmap(img):
    """Chuyển ảnh BGR hoặc xám (numpy) sang QPixmap."""
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class ImageLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)  # tọa độ trên ảnh gốc

    def __init__(self, title):
        super().__init__(title)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setObjectName("imageView")
        self._img = None

    def show_image(self, img):
        self._img = img
        self._render()

    def _render(self):
        if self._img is None:
            return
        pix = to_qpixmap(self._img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def mousePressEvent(self, e):
        pix = self.pixmap()
        if pix is None or self._img is None:
            return
        off_x = (self.width() - pix.width()) / 2
        off_y = (self.height() - pix.height()) / 2
        px, py = e.position().x() - off_x, e.position().y() - off_y
        if 0 <= px < pix.width() and 0 <= py < pix.height():
            ih, iw = self._img.shape[:2]
            x = min(int(px * iw / pix.width()), iw - 1)
            y = min(int(py * ih / pix.height()), ih - 1)
            self.clicked.emit(x, y)
