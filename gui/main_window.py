import cv2
from PyQt5 import QtWidgets, QtCore
from gui.image_view import ImageLabel

# Khi TV1 xong: đổi thành `from hsv_core import segment_hsv, suggest_range`
from hsv_core_mock import segment_hsv, suggest_range


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV Segmentation")
        self.img = None
        self.cap = None
        self.last_result = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.views = [ImageLabel(t) for t in ("Ảnh gốc", "Mask", "Kết quả")]
        self.views[0].clicked.connect(self.on_pick_color)

        spec = [("H min", 0, 179, 0), ("H max", 0, 179, 179),
                ("S min", 0, 255, 0), ("S max", 0, 255, 255),
                ("V min", 0, 255, 0), ("V max", 0, 255, 255)]
        self.sliders = {}
        form = QtWidgets.QFormLayout()
        for name, lo, hi, df in spec:
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(df)
            s.valueChanged.connect(self.update_result)
            lbl = QtWidgets.QLabel(str(df))
            s.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(s)
            row.addWidget(lbl)
            form.addRow(name, row)
            self.sliders[name] = (s, df)

        buttons = QtWidgets.QHBoxLayout()
        for text, fn in (("Mở ảnh", self.open_image), ("Mở video", self.open_video),
                         ("Lưu kết quả", self.save_result), ("Reset", self.reset)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            buttons.addWidget(b)

        images = QtWidgets.QHBoxLayout()
        for v in self.views:
            images.addWidget(v)

        root = QtWidgets.QVBoxLayout()
        root.addLayout(images)
        root.addLayout(form)
        root.addLayout(buttons)
        w = QtWidgets.QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

    def get_range(self):
        v = {k: s.value() for k, (s, _) in self.sliders.items()}
        return ((v["H min"], v["S min"], v["V min"]),
                (v["H max"], v["S max"], v["V max"]))

    def update_result(self):
        if self.img is None:
            return
        lower, upper = self.get_range()
        mask, result = segment_hsv(self.img, lower, upper)
        self.views[0].show_image(self.img)
        self.views[1].show_image(mask)
        self.views[2].show_image(result)
        self.last_result = result

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.stop_video()
            self.img = cv2.imread(path)
            self.update_result()

    def save_result(self):
        if self.last_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu", "result.png", "PNG (*.png);;JPG (*.jpg)")
        if path:
            cv2.imwrite(path, self.last_result)

    def reset(self):
        for s, df in self.sliders.values():
            s.setValue(df)

    def on_pick_color(self, x, y):
        if self.img is None:
            return
        lower, upper = suggest_range(self.img, x, y)
        for key, val in zip(("H min", "S min", "V min"), lower):
            self.sliders[key][0].setValue(val)
        for key, val in zip(("H max", "S max", "V max"), upper):
            self.sliders[key][0].setValue(val)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video (*.mp4 *.avi *.mov)")
        if path:
            self.cap = cv2.VideoCapture(path)
            self.timer.start(30)

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.stop_video()
            return
        self.img = frame
        self.update_result()

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
