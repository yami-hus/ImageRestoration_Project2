"""Bản giả tạm cho TV2 dùng đến khi TV1 xong hsv_core.py."""
import cv2
import numpy as np


def segment_hsv(img_bgr, lower, upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return mask, result


def suggest_range(img_bgr, x, y, radius=5):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [int(c) for c in hsv[y, x]]
    lower = (max(h - 10, 0), max(s - 60, 0), max(v - 60, 0))
    upper = (min(h + 10, 179), min(s + 60, 255), min(v + 60, 255))
    return lower, upper
