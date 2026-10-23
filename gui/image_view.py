import cv2
from PyQt5 import QtWidgets, QtGui, QtCore


def to_qpixmap(img):
    """Chuyển ảnh BGR hoặc xám (numpy) sang QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class ImageLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int)  # tọa độ trên ảnh gốc

    def __init__(self, title):
        super().__init__(title)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background:#222; color:#aaa;")
        self.img_size = None  # (w, h) của ảnh gốc

    def show_image(self, img):
        self.img_size = (img.shape[1], img.shape[0])
        pix = to_qpixmap(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)

    def mousePressEvent(self, e):
        pix = self.pixmap()
        if pix is None or self.img_size is None:
            return
        off_x = (self.width() - pix.width()) / 2
        off_y = (self.height() - pix.height()) / 2
        px, py = e.x() - off_x, e.y() - off_y
        if 0 <= px < pix.width() and 0 <= py < pix.height():
            x = int(px * self.img_size[0] / pix.width())
            y = int(py * self.img_size[1] / pix.height())
            self.clicked.emit(x, y)
