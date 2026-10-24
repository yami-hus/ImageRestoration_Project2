"""TV3: sinh ảnh tổng hợp có đáp án biết trước."""
import os
import cv2
import numpy as np

SIZE = 80
# (tên, màu BGR, H theo thang OpenCV)
PATCHES = [
    ("red_low",  (0, 0, 255),  0),
    ("red_high", (30, 0, 255), 176),
    ("green",    (0, 255, 0),  60),
    ("blue",     (255, 0, 0),  120),
]


def make_synthetic():
    """Trả (ảnh BGR 100x400, dict tên -> mask 0/255 của từng ô)."""
    img = np.full((100, 400, 3), 128, np.uint8)  # nền xám, S = 0
    masks = {}
    for i, (name, bgr, _) in enumerate(PATCHES):
        x = 10 + i * 100
        img[10:10 + SIZE, x:x + SIZE] = bgr
        m = np.zeros(img.shape[:2], np.uint8)
        m[10:10 + SIZE, x:x + SIZE] = 255
        masks[name] = m
    return img, masks


if __name__ == "__main__":
    img, masks = make_synthetic()
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)
    cv2.imwrite("data/images/synthetic_01.png", img)
    # mask chuẩn = hai ô đỏ (đỏ vòng Hue)
    gt = cv2.bitwise_or(masks["red_low"], masks["red_high"])
    cv2.imwrite("data/masks/synthetic_01.png", gt)
    print("Đã tạo synthetic_01.png")