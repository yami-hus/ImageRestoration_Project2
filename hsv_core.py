from dataclasses import dataclass

import cv2
import numpy as np

H_MAX = 179
SV_MAX = 255


# %% [THAM SỐ PHÂN ĐOẠN]
@dataclass
class HSVParams:
    h_min: int
    h_max: int
    s_min: int = 0
    s_max: int = SV_MAX
    v_min: int = 0
    v_max: int = SV_MAX
    morph_kernel: int = 0
    morph_iter: int = 1


def _check_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} phải là số nguyên, nhận được {value!r}")


# %% [KIỂM TRA THAM SỐ]
def validate_hsv_params(params):
    if not isinstance(params, HSVParams):
        raise ValueError("params phải là một HSVParams")

    for name in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max",
                 "morph_kernel", "morph_iter"):
        _check_int(name, getattr(params, name))

    for name in ("h_min", "h_max"):
        v = getattr(params, name)
        if not 0 <= v <= H_MAX:
            raise ValueError(f"{name} phải trong khoảng 0-{H_MAX}, nhận được {v}")

    for name in ("s_min", "s_max", "v_min", "v_max"):
        v = getattr(params, name)
        if not 0 <= v <= SV_MAX:
            raise ValueError(f"{name} phải trong khoảng 0-{SV_MAX}, nhận được {v}")
    if params.s_min > params.s_max:
        raise ValueError(f"s_min ({params.s_min}) không được lớn hơn s_max ({params.s_max})")
    if params.v_min > params.v_max:
        raise ValueError(f"v_min ({params.v_min}) không được lớn hơn v_max ({params.v_max})")

    k = params.morph_kernel
    if k < 0:
        raise ValueError(f"morph_kernel không được âm, nhận được {k}")
    if k != 0 and k % 2 == 0:
        raise ValueError(f"morph_kernel phải là số lẻ (hoặc 0 để tắt), nhận được {k}")
    if params.morph_iter < 1:
        raise ValueError(f"morph_iter phải >= 1, nhận được {params.morph_iter}")


def _check_bgr_image(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a valid numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Ảnh phải là BGR 3 kênh (HxWx3), nhận được shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Ảnh phải là uint8, nhận được {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Ảnh rỗng (kích thước 0)")


# %% [TẠO MASK HSV]
def build_hsv_mask(image, params):
    _check_bgr_image(image)
    validate_hsv_params(params)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_sv = (params.s_min, params.v_min)
    upper_sv = (params.s_max, params.v_max)

    if params.h_min <= params.h_max:
        lower = np.array([params.h_min, *lower_sv], dtype=np.uint8)
        upper = np.array([params.h_max, *upper_sv], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    low_a = np.array([params.h_min, *lower_sv], dtype=np.uint8)
    up_a = np.array([H_MAX, *upper_sv], dtype=np.uint8)
    low_b = np.array([0, *lower_sv], dtype=np.uint8)
    up_b = np.array([params.h_max, *upper_sv], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, low_a, up_a),
                          cv2.inRange(hsv, low_b, up_b))


# %% [MORPHOLOGY]
def apply_morphology(mask, kernel_size, iterations=1):
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask phải là mảng numpy 1 kênh (HxW)")
    if kernel_size < 0 or (kernel_size != 0 and kernel_size % 2 == 0):
        raise ValueError(f"kernel_size phải là 0 hoặc số lẻ dương, nhận được {kernel_size}")
    if iterations < 1:
        raise ValueError(f"iterations phải >= 1, nhận được {iterations}")
    if kernel_size == 0:
        return mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# %% [PHÂN ĐOẠN HSV]
def apply_hsv_segmentation(image, params):
    mask = build_hsv_mask(image, params)
    if params.morph_kernel > 0:
        mask = apply_morphology(mask, params.morph_kernel, params.morph_iter)
    result = np.zeros_like(image)
    cv2.copyTo(image, mask, result)
    return mask, result


# %% [OVERLAY]
def make_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    _check_bgr_image(image)
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask phải là mảng numpy 1 kênh (HxW)")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} không cùng kích thước ảnh {image.shape[:2]}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha phải trong khoảng 0-1, nhận được {alpha}")
    if len(color) != 3 or any(not 0 <= c <= 255 for c in color):
        raise ValueError(f"color phải gồm 3 giá trị BGR trong 0-255, nhận được {color}")

    overlay = image.copy()
    region = mask > 0
    color_arr = np.array(color, dtype=np.float32)
    blended = image[region].astype(np.float32) * (1 - alpha) + color_arr * alpha
    overlay[region] = np.clip(blended, 0, 255).astype(np.uint8)
    return overlay


# %% [ĐỀ XUẤT KHOẢNG HSV TỪ ĐIỂM BẤM - MỞ RỘNG]
def suggest_hsv_range(image, x, y, tolerance=(10, 60, 60), patch=5):
    _check_bgr_image(image)
    h_img, w_img = image.shape[:2]
    if not (0 <= x < w_img and 0 <= y < h_img):
        raise ValueError(f"Điểm ({x}, {y}) nằm ngoài ảnh {w_img}x{h_img}")
    if isinstance(tolerance, (int, np.integer)):
        tolerance = (tolerance, tolerance, tolerance)
    if len(tolerance) != 3 or any(t < 0 for t in tolerance):
        raise ValueError(f"tolerance phải là 3 số không âm, nhận được {tolerance}")
    if patch < 1 or patch % 2 == 0:
        raise ValueError(f"patch phải là số lẻ dương, nhận được {patch}")

    r = patch // 2
    y0, y1 = max(0, y - r), min(h_img, y + r + 1)
    x0, x1 = max(0, x - r), min(w_img, x + r + 1)
    hsv_patch = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (int(c) for c in np.median(hsv_patch, axis=0))
    dh, ds, dv = (int(t) for t in tolerance)

    if 2 * dh + 1 >= H_MAX + 1:
        h_min, h_max = 0, H_MAX
    else:
        h_min = (h - dh) % (H_MAX + 1)
        h_max = (h + dh) % (H_MAX + 1)

    return HSVParams(
        h_min=h_min, h_max=h_max,
        s_min=max(0, s - ds), s_max=min(SV_MAX, s + ds),
        v_min=max(0, v - dv), v_max=min(SV_MAX, v + dv),
    )


# %% [THỐNG KÊ MASK - ĐÁNH GIÁ NHANH KHÔNG CẦN MASK CHUẨN]
def mask_statistics(mask):
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask phải là mảng numpy 1 kênh (HxW)")
    if mask.size == 0:
        raise ValueError("mask rỗng (kích thước 0)")

    binary = (mask > 0).astype(np.uint8)
    count = int(np.count_nonzero(binary))
    total = int(binary.size)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {
        "pixel_count": count,
        "total_pixels": total,
        "coverage": count / total,
        "num_components": int(num_labels - 1),
        "largest_area": int(areas.max()) if areas.size else 0,
        "areas": sorted((int(a) for a in areas), reverse=True),
    }


def segment_hsv(image, lower, upper):
    params = HSVParams(
        h_min=int(lower[0]), h_max=int(upper[0]),
        s_min=int(lower[1]), s_max=int(upper[1]),
        v_min=int(lower[2]), v_max=int(upper[2]),
    )
    return apply_hsv_segmentation(image, params)


def suggest_range(image, x, y, radius=5, tolerance=(10, 60, 60)):
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError(f"radius phải là số nguyên >= 1, nhận được {radius!r}")
    p = suggest_hsv_range(image, x, y, tolerance=tolerance, patch=2 * int(radius) + 1)
    return (p.h_min, p.s_min, p.v_min), (p.h_max, p.s_max, p.v_max)


def remove_small_regions(mask, min_area):
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask phải là mảng numpy 1 kênh (HxW)")
    if min_area < 0:
        raise ValueError(f"min_area không được âm, nhận được {min_area}")
    if min_area == 0:
        return mask.copy()
    binary = (mask > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                      # nhãn 0 là nền
    return keep[labels].astype(np.uint8) * 255


def clean_mask(mask, kernel_size, min_area=0):
    cleaned = apply_morphology(mask, kernel_size, 1)
    return remove_small_regions(cleaned, min_area)


def region_stats(mask):
    s = mask_statistics(mask)
    return {"coverage": s["coverage"], "n_regions": s["num_components"], "areas": s["areas"]}
