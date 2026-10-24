
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hsv_core as core          # noqa: E402
import seg_metrics as metrics    # noqa: E402

IMG_DIR = os.path.join(ROOT, "data", "images")
MASK_DIR = os.path.join(ROOT, "data", "masks")
OUT_DIR = os.path.join(ROOT, "output", "eval")
PRESET_FILE = os.path.join(ROOT, "presets.json")
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_SIDE = 1000          # thu nhỏ ảnh quá lớn cho nhanh
CLEAN_KERNEL = 3
CLEAN_MIN_AREA = 50


def read_image(path, flag=cv2.IMREAD_COLOR):
    """Đọc ảnh an toàn với đường dẫn có dấu tiếng Việt. Trả None nếu lỗi."""
    data = np.fromfile(path, np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flag)


def shrink(img, interp):
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)


def load_presets():
    with open(PRESET_FILE, encoding="utf-8") as f:
        items = json.load(f)
    return [(p["name"], tuple(p["lower"]), tuple(p["upper"])) for p in items]


def evaluate_image(path, presets):
    """Trả (ảnh, mask chuẩn hoặc None, danh sách kết quả từng preset)."""
    img = read_image(path)
    if img is None:
        print(f"  [BỎ QUA] không đọc được: {os.path.basename(path)}")
        return None, None, []
    img = shrink(img, cv2.INTER_AREA)

    stem = os.path.splitext(os.path.basename(path))[0]
    gt = None
    gt_path = os.path.join(MASK_DIR, stem + ".png")
    if os.path.exists(gt_path):
        raw = read_image(gt_path, cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            raw = cv2.resize(raw, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            gt = ((raw > 127) * 255).astype(np.uint8)

    rows = []
    for name, lower, upper in presets:
        mask, result = core.segment_hsv(img, lower, upper)
        cleaned = core.clean_mask(mask, kernel_size=CLEAN_KERNEL,
                                  min_area=CLEAN_MIN_AREA)
        row = {
            "preset": name, "mask": mask, "result": result, "cleaned": cleaned,
            "coverage": float(np.count_nonzero(mask)) / mask.size,
            "coverage_clean": float(np.count_nonzero(cleaned)) / cleaned.size,
            "iou": None, "dice": None, "iou_clean": None, "dice_clean": None,
        }
        if gt is not None:
            row["iou"] = metrics.iou(mask, gt)
            row["dice"] = metrics.dice(mask, gt)
            row["iou_clean"] = metrics.iou(cleaned, gt)
            row["dice_clean"] = metrics.dice(cleaned, gt)
        rows.append(row)
    return img, gt, rows


def save_grid(stem, img, gt, rows):
    """Hàng trên: ảnh gốc + kết quả từng preset. Hàng dưới: mask chuẩn + mask."""
    n = len(rows)
    fig, axes = plt.subplots(2, n + 1, figsize=(2.6 * (n + 1), 5.4))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[0][0].imshow(rgb)
    axes[0][0].set_title("Ảnh gốc", fontsize=9)
    if gt is not None:
        axes[1][0].imshow(gt, cmap="gray", vmin=0, vmax=255)
        axes[1][0].set_title("Mask chuẩn", fontsize=9)

    for j, r in enumerate(rows, start=1):
        axes[0][j].imshow(cv2.cvtColor(r["result"], cv2.COLOR_BGR2RGB))
        axes[0][j].set_title(r["preset"], fontsize=9)
        axes[1][j].imshow(r["mask"], cmap="gray", vmin=0, vmax=255)
        sub = f"phủ {r['coverage'] * 100:.1f}%"
        if r["iou"] is not None:
            sub += f"\nIoU {r['iou']:.2f}  Dice {r['dice']:.2f}"
        axes[1][j].set_title(sub, fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"grid_{stem}.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    presets = load_presets()
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(EXTS))
    print(f"{len(files)} ảnh, {len(presets)} preset\n")

    csv_rows = []
    best_lines = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        print(f"- {fname}")
        img, gt, rows = evaluate_image(os.path.join(IMG_DIR, fname), presets)
        if img is None:
            continue
        save_grid(stem, img, gt, rows)

        for r in rows:
            csv_rows.append({
                "anh": fname, "preset": r["preset"],
                "co_mask_chuan": "x" if gt is not None else "",
                "phu_%": round(r["coverage"] * 100, 2),
                "phu_sau_clean_%": round(r["coverage_clean"] * 100, 2),
                "iou": "" if r["iou"] is None else round(r["iou"], 4),
                "dice": "" if r["dice"] is None else round(r["dice"], 4),
                "iou_sau_clean": "" if r["iou_clean"] is None else round(r["iou_clean"], 4),
                "dice_sau_clean": "" if r["dice_clean"] is None else round(r["dice_clean"], 4),
                "danh_gia_tay": "",   # điền: dat / mot phan / that bai + lý do
            })

        if gt is not None:
            best = max(rows, key=lambda r: r["iou"])
            best_lines.append(
                f"{fname}: tốt nhất là '{best['preset']}' "
                f"(IoU {best['iou']:.3f}, Dice {best['dice']:.3f}; "
                f"sau clean_mask IoU {best['iou_clean']:.3f})")

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n=== Ảnh có mask chuẩn: preset cho IoU cao nhất ===")
    print("\n".join(best_lines) if best_lines else "(chưa có mask chuẩn nào)")
    print(f"\nĐã lưu: {csv_path}")
    print(f"Hình so sánh: {OUT_DIR}")


if __name__ == "__main__":
    main()