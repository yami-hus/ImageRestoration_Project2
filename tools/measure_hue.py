
import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "data", "images")
PREFIXES = tuple(f"img_0{i}" for i in range(1, 9))

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith(PREFIXES):
        continue
    data = np.fromfile(os.path.join(IMG_DIR, fname), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{fname}: không đọc được")
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sel = (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)   # bỏ nền trắng, bóng xám
    h = hsv[..., 0][sel]
    if h.size == 0:
        print(f"{fname}: không có pixel đủ bão hòa")
        continue
    p10, p50, p90 = np.percentile(h, [10, 50, 90]).astype(int)
    s_p10 = int(np.percentile(hsv[..., 1][sel], 10))
    v_p10 = int(np.percentile(hsv[..., 2][sel], 10))
    print(f"{fname}: H p10={p10} p50={p50} p90={p90} | S p10={s_p10} V p10={v_p10}")