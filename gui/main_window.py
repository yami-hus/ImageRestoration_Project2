import os
import json
import cv2
from PySide6 import QtWidgets, QtCore

from gui.image_view import ImageLabel
from gui.widgets import NoWheelSlider, NoWheelComboBox, WheelGuard
from gui.histogram import HueHistogram, compute_h_hist
from gui.backend import segment_hsv, suggest_range, clean_mask, region_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET_PATH = os.path.join(ROOT, "presets.json")

# (tên, min, max, mặc định)
SLIDERS = [("H min", 0, 179, 0), ("H max", 0, 179, 179),
           ("S min", 0, 255, 0), ("S max", 0, 255, 255),
           ("V min", 0, 255, 0), ("V max", 0, 255, 255)]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSV Segmentation")
        self.img = None
        self.cap = None
        self.video_path = None
        self.last_mask = None
        self.last_result = None
        self.presets = []

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self._build_ui()
        self.load_presets(PRESET_PATH)
        self.statusBar().showMessage("Bắt đầu bằng cách mở một ảnh hoặc video.")

    # ================= UI =================
    @staticmethod
    def _card(title, inner, hint=None):
        """Khung (thẻ) có tiêu đề; inner là layout hoặc widget."""
        f = QtWidgets.QFrame()
        f.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(f)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(title)
        t.setObjectName("cardTitle")
        head.addWidget(t)
        if hint:
            h = QtWidgets.QLabel(hint)
            h.setObjectName("hint")
            head.addWidget(h)
        head.addStretch(1)
        lay.addLayout(head)
        if isinstance(inner, QtWidgets.QLayout):
            lay.addLayout(inner, 1)
        else:
            lay.addWidget(inner, 1)
        return f

    @staticmethod
    def _button(text, fn, variant=None):
        b = QtWidgets.QPushButton(text)
        if variant:
            b.setProperty("variant", variant)
        b.setCursor(QtCore.Qt.PointingHandCursor)
        b.clicked.connect(fn)
        return b

    def _build_ui(self):
        # --- tiêu đề ---
        title = QtWidgets.QLabel("HSV Segmentation")
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel("Tách vùng theo màu trong không gian HSV")
        subtitle.setObjectName("subtitle")
        tcol = QtWidgets.QVBoxLayout()
        tcol.setSpacing(0)
        tcol.addWidget(title)
        tcol.addWidget(subtitle)
        self.lbl_source = QtWidgets.QLabel("Chưa mở ảnh hoặc video")
        self.lbl_source.setObjectName("source")
        header = QtWidgets.QHBoxLayout()
        header.addLayout(tcol)
        header.addStretch(1)
        header.addWidget(self.lbl_source)

        # --- 3 khung ảnh ---
        self.views = [ImageLabel(t) for t in ("Ảnh gốc", "Mask", "Kết quả")]
        self.views[0].clicked.connect(self.on_pick_color)
        images = QtWidgets.QHBoxLayout()
        images.setSpacing(12)
        for cap, hint, v in (("Ảnh gốc", "Click vào ảnh để chọn màu", self.views[0]),
                             ("Mask", None, self.views[1]),
                             ("Kết quả", None, self.views[2])):
            images.addWidget(self._card(cap, v, hint), 1)

        # --- thanh trượt ---
        self.sliders, self.value_labels = {}, {}
        sl_grid = QtWidgets.QGridLayout()
        sl_grid.setHorizontalSpacing(10)
        sl_grid.setVerticalSpacing(12)
        for row, (name, lo, hi, df) in enumerate(SLIDERS):
            nm = QtWidgets.QLabel(name)
            nm.setObjectName("rowName")
            s = NoWheelSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(df)
            s.setProperty("kind", name[0].lower())   # h / s / v -> tô dải màu
            lbl = QtWidgets.QLabel(str(df))
            lbl.setObjectName("valuePill")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            s.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
            s.valueChanged.connect(lambda v, n=name: self._clamp_sv(n))
            s.valueChanged.connect(self.update_result)
            sl_grid.addWidget(nm, row, 0)
            sl_grid.addWidget(s, row, 1)
            sl_grid.addWidget(lbl, row, 2)
            self.sliders[name] = s
            self.value_labels[name] = lbl
        sl_grid.setColumnStretch(1, 1)
        note = QtWidgets.QLabel("H min lớn hơn H max nghĩa là màu đỏ (vòng qua 179 về 0)")
        note.setObjectName("hint")
        note.setWordWrap(True)
        thr_lay = QtWidgets.QVBoxLayout()
        thr_lay.addLayout(sl_grid)
        thr_lay.addWidget(note)
        thr_lay.addStretch(1)
        card_thr = self._card("Ngưỡng HSV", thr_lay)

        # --- preset + chọn màu ---
        self.combo = NoWheelComboBox()
        self.combo.activated.connect(self.on_preset_selected)
        self.spin_radius = QtWidgets.QSpinBox()
        self.spin_radius.setRange(1, 30)
        self.spin_radius.setValue(5)
        self.spin_radius.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        pl = QtWidgets.QGridLayout()
        pl.setVerticalSpacing(8)
        pl.addWidget(self.combo, 0, 0, 1, 2)
        pl.addWidget(self._button("Lưu preset", self.save_preset), 1, 0)
        pl.addWidget(self._button("Nạp file", self.load_preset_file), 1, 1)
        rl = QtWidgets.QLabel("Bán kính lấy mẫu (px)")
        rl.setObjectName("rowName")
        pl.addWidget(rl, 2, 0)
        pl.addWidget(self.spin_radius, 2, 1)
        card_preset = self._card("Preset và chọn màu", pl)

        # --- hậu xử lý ---
        self.chk_post = QtWidgets.QCheckBox("Làm sạch mask")
        self.chk_post.toggled.connect(self.update_result)
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setRange(1, 31)
        self.spin_k.setSingleStep(2)
        self.spin_k.setValue(3)
        self.spin_k.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin_k.valueChanged.connect(self.update_result)
        self.spin_area = QtWidgets.QSpinBox()
        self.spin_area.setRange(0, 100000)
        self.spin_area.setValue(50)
        self.spin_area.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spin_area.valueChanged.connect(self.update_result)
        pf = QtWidgets.QGridLayout()
        pf.setVerticalSpacing(8)
        pf.addWidget(self.chk_post, 0, 0, 1, 2)
        for r, (text, w) in enumerate((("Kernel", self.spin_k), ("Diện tích tối thiểu (px)", self.spin_area)), 1):
            lb = QtWidgets.QLabel(text)
            lb.setObjectName("rowName")
            pf.addWidget(lb, r, 0)
            pf.addWidget(w, r, 1)
        card_post = self._card("Hậu xử lý", pf)

        # --- histogram + thống kê ---
        self.hist_widget = HueHistogram()
        card_hist = self._card("Histogram kênh H", self.hist_widget)

        self.m_cov, self.m_n, self.m_big = (QtWidgets.QLabel("-") for _ in range(3))
        mg = QtWidgets.QGridLayout()
        for col, (w, cap) in enumerate(((self.m_cov, "Diện tích"), (self.m_n, "Số vùng"),
                                        (self.m_big, "Vùng lớn nhất"))):
            w.setObjectName("metricValue")
            c = QtWidgets.QLabel(cap)
            c.setObjectName("metricCaption")
            mg.addWidget(w, 0, col)
            mg.addWidget(c, 1, col)
        self.lbl_thr = QtWidgets.QLabel("")
        self.lbl_thr.setObjectName("hint")
        self.lbl_thr.setWordWrap(True)
        st = QtWidgets.QVBoxLayout()
        st.addLayout(mg)
        st.addWidget(self.lbl_thr)
        card_stats = self._card("Thống kê", st)

        # --- thanh nút ---
        self.btn_play = self._button("Phát", self.toggle_play)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self._button("Mở ảnh", self.open_image, "primary"))
        actions.addWidget(self._button("Mở video", self.open_video))
        actions.addWidget(self.btn_play)
        actions.addStretch(1)
        actions.addWidget(self._button("Lưu kết quả", self.save_result))
        actions.addWidget(self._button("Lưu mask", self.save_mask))
        actions.addWidget(self._button("Lưu video", self.save_video))
        actions.addWidget(self._button("Đặt lại", self.reset, "danger"))

        # ô số: chỉ nhận lăn chuột khi đang được chọn
        self._wheel_guard = WheelGuard(lambda: getattr(self, "_scroll", None))
        for sp in (self.spin_radius, self.spin_k, self.spin_area):
            sp.installEventFilter(self._wheel_guard)
            sp.setFocusPolicy(QtCore.Qt.StrongFocus)

        # chiều cao tối thiểu: cửa sổ nhỏ hơn mức này thì hiện thanh cuộn
        for card, h in ((card_thr, 280), (card_preset, 175), (card_post, 175),
                        (card_hist, 140), (card_stats, 160)):
            card.setMinimumHeight(h)

        # --- ghép layout ---
        mid = QtWidgets.QVBoxLayout()
        mid.setSpacing(12)
        mid.addWidget(card_preset)
        mid.addWidget(card_post)
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(card_hist, 1)
        right.addWidget(card_stats)
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(card_thr, 4)
        bottom.addLayout(mid, 3)
        bottom.addLayout(right, 3)

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(18, 14, 18, 10)
        root.setSpacing(12)
        root.addLayout(header)
        root.addLayout(images, 3)
        root.addLayout(bottom, 2)
        root.addLayout(actions)
        w = QtWidgets.QWidget()
        w.setObjectName("root")
        w.setLayout(root)
        w.setMinimumWidth(1000)
        scroll = self._scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(w)
        self.setCentralWidget(scroll)

    # ================= tham số =================
    def get_range(self):
        v = {k: s.value() for k, s in self.sliders.items()}
        return ((v["H min"], v["S min"], v["V min"]),
                (v["H max"], v["S max"], v["V max"]))

    def set_range(self, lower, upper):
        """Gán 6 thanh trượt một lượt, chỉ cập nhật kết quả 1 lần."""
        vals = dict(zip(("H min", "S min", "V min"), lower))
        vals.update(zip(("H max", "S max", "V max"), upper))
        for k, v in vals.items():
            s = self.sliders[k]
            s.blockSignals(True)
            s.setValue(int(v))
            s.blockSignals(False)
            self.value_labels[k].setText(str(int(v)))
        self.update_result()

    def _clamp_sv(self, changed):
        """S và V: min không được vượt max (riêng H được phép để vòng màu đỏ)."""
        ch = changed[0]
        if ch not in "SV":
            return
        lo, hi = f"{ch} min", f"{ch} max"
        if self.sliders[lo].value() > self.sliders[hi].value():
            other = hi if changed == lo else lo
            val = self.sliders[changed].value()
            self.sliders[other].blockSignals(True)
            self.sliders[other].setValue(val)
            self.sliders[other].blockSignals(False)
            self.value_labels[other].setText(str(val))

    def reset(self):
        self.combo.setCurrentIndex(0)
        self.set_range((0, 0, 0), (179, 255, 255))

    # ================= xử lý =================
    def _process(self, img):
        lower, upper = self.get_range()
        mask, _ = segment_hsv(img, lower, upper)
        if self.chk_post.isChecked():
            k = self.spin_k.value()
            k = k if k % 2 == 1 else k + 1
            mask = clean_mask(mask, k, self.spin_area.value())
        result = cv2.bitwise_and(img, img, mask=mask)
        return mask, result

    def update_result(self, *_):
        if self.img is None:
            return
        mask, result = self._process(self.img)
        self.last_mask, self.last_result = mask, result
        self.views[0].show_image(self.img)
        self.views[1].show_image(mask)
        self.views[2].show_image(result)

        lower, upper = self.get_range()
        self.hist_widget.set_data(compute_h_hist(self.img))
        self.hist_widget.set_range(lower[0], upper[0])

        st = region_stats(mask)
        largest = max(st["areas"]) if st["areas"] else 0
        self.m_cov.setText(f"{st['coverage'] * 100:.1f}%")
        self.m_n.setText(str(st["n_regions"]))
        self.m_big.setText(f"{largest:,}")
        self.lbl_thr.setText(f"Ngưỡng dưới {lower}, ngưỡng trên {upper}")

    # ================= ảnh =================
    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không đọc được ảnh này.")
            return
        self.stop_video(release=True)
        self.video_path = None
        self.img = img
        self.update_result()
        self.lbl_source.setText(f"{os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
        self.statusBar().showMessage("Đã mở ảnh. Click vào ảnh gốc để chọn màu.")

    def _save_image(self, img, default_name):
        if img is None:
            QtWidgets.QMessageBox.information(self, "Thông báo", "Chưa có kết quả để lưu.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu", default_name, "PNG (*.png);;JPG (*.jpg)")
        if path:
            cv2.imwrite(path, img)
            self.statusBar().showMessage(f"Đã lưu: {path}")

    def save_result(self):
        self._save_image(self.last_result, "result.png")

    def save_mask(self):
        self._save_image(self.last_mask, "mask.png")

    # ================= eyedropper =================
    def on_pick_color(self, x, y):
        if self.img is None:
            return
        lower, upper = suggest_range(self.img, x, y, self.spin_radius.value())
        self.combo.setCurrentIndex(0)
        self.set_range(lower, upper)

    # ================= preset =================
    def refresh_combo(self):
        self.combo.clear()
        self.combo.addItem("-- Chọn preset --")
        for p in self.presets:
            self.combo.addItem(p["name"])

    def load_presets(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                self.presets = json.load(f)
        except (OSError, ValueError):
            self.presets = []
        self.refresh_combo()

    def load_preset_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Nạp preset", "", "JSON (*.json)")
        if path:
            self.load_presets(path)

    def on_preset_selected(self, idx):
        if idx <= 0:
            return
        p = self.presets[idx - 1]
        self.set_range(p["lower"], p["upper"])

    def save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Lưu preset", "Tên preset:")
        name = name.strip()
        if not ok or not name:
            return
        lower, upper = self.get_range()
        item = {"name": name, "lower": list(lower), "upper": list(upper)}
        for i, p in enumerate(self.presets):
            if p["name"] == name:
                self.presets[i] = item
                break
        else:
            self.presets.append(item)
        with open(PRESET_PATH, "w", encoding="utf-8") as f:
            json.dump(self.presets, f, ensure_ascii=False, indent=2)
        self.refresh_combo()
        self.combo.setCurrentIndex(self.combo.findText(name))
        self.statusBar().showMessage(f"Đã lưu preset '{name}'")

    # ================= video =================
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không mở được video này.")
            return
        self.stop_video(release=True)
        self.cap, self.video_path = cap, path
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.timer.setInterval(max(int(1000 / fps), 1))
        self.timer.start()
        self.btn_play.setText("Dừng")
        self.lbl_source.setText(f"{os.path.basename(path)} ({fps:.0f} fps)")
        self.statusBar().showMessage("Đang phát video.")

    def toggle_play(self):
        if self.cap is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Phát")
        else:
            self.timer.start()
            self.btn_play.setText("Dừng")

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:  # hết video: dừng và tua về đầu
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.btn_play.setText("Phát")
            return
        self.img = frame
        self.update_result()

    def stop_video(self, release=False):
        self.timer.stop()
        self.btn_play.setText("Phát")
        if release and self.cap:
            self.cap.release()
            self.cap = None

    def save_video(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "Thông báo", "Hãy mở video trước.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu video kết quả", "result.mp4", "MP4 (*.mp4);;AVI (*.avi)")
        if not out_path:
            return
        was_playing = self.timer.isActive()
        self.timer.stop()

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        dlg = QtWidgets.QProgressDialog("Đang xử lý video...", "Hủy", 0, total, self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        writer, i = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, result = self._process(frame)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(result)
            i += 1
            dlg.setValue(min(i, total))
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                break
        cap.release()
        if writer:
            writer.release()
        dlg.close()
        self.statusBar().showMessage(f"Đã lưu video ({i} khung hình): {out_path}")
        if was_playing:
            self.timer.start()

    def closeEvent(self, e):
        self.stop_video(release=True)
        super().closeEvent(e)
