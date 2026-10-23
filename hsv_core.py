"""TV1: thuật toán lõi. Chữ ký hàm xem INTERFACE.md."""
import numpy as np


def bgr_to_hsv(img_bgr):
    raise NotImplementedError


def segment_hsv(img_bgr, lower, upper):
    raise NotImplementedError


def suggest_range(img_bgr, x, y, radius=5):
    raise NotImplementedError


def clean_mask(mask, kernel_size=3, min_area=50):
    raise NotImplementedError


def region_stats(mask):
    raise NotImplementedError


def iou(pred, gt):
    raise NotImplementedError


def dice(pred, gt):
    raise NotImplementedError
