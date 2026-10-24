"""Chỉ dùng cho ảnh có mask chuẩn."""
import numpy as np


def _check_pair(pred, gt):
    for name, m in (("pred", pred), ("gt", gt)):
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            raise ValueError(f"{name} phải là mảng 2 chiều (H, W)")
    if pred.shape != gt.shape:
        raise ValueError(f"Kích thước không khớp: {pred.shape} và {gt.shape}")
    return pred > 0, gt > 0


def iou(pred, gt):
    """IoU giữa hai mask nhị phân. Cả hai rỗng thì trả 1.0."""
    p, g = _check_pair(pred, gt)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def dice(pred, gt):
    """Dice giữa hai mask nhị phân. Cả hai rỗng thì trả 1.0."""
    p, g = _check_pair(pred, gt)
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / total)