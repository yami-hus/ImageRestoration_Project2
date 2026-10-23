"""Bản giả tạm cho TV2 dùng đến khi TV1 xong hsv_core.py (cùng chữ ký hàm)."""
import cv2
import numpy as np


def segment_hsv(img_bgr, lower, upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo, up = list(lower), list(upper)
    if lo[0] <= up[0]:
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(up, np.uint8))
    else:  # màu đỏ vòng: H nằm ở hai đầu thang
        m1 = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                         np.array([179, up[1], up[2]], np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                         np.array([up[0], up[1], up[2]], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return mask, result


def suggest_range(img_bgr, x, y, radius=5):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]
    patch = hsv[max(y - radius, 0):min(y + radius + 1, h_img),
                max(x - radius, 0):min(x + radius + 1, w_img)].reshape(-1, 3)
    h, s, v = [int(c) for c in np.median(patch, axis=0)]
    lower = (max(h - 10, 0), max(s - 60, 0), max(v - 60, 0))
    upper = (min(h + 10, 179), min(s + 60, 255), min(v + 60, 255))
    return lower, upper


def clean_mask(mask, kernel_size=3, min_area=50):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def region_stats(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    return {"coverage": float(np.count_nonzero(mask)) / mask.size,
            "n_regions": len(areas), "areas": areas}
