# Hợp đồng giao diện hàm (cả nhóm tuân thủ)

## Quy ước dữ liệu
- Ảnh: BGR, `uint8`, shape (H, W, 3)
- Mask: `uint8`, shape (H, W), giá trị 0 hoặc 255
- `lower`, `upper`: tuple 3 số nguyên (H, S, V) theo thang OpenCV (H: 0-179, S/V: 0-255)

## TV1 - hsv_core.py
```python
def segment_hsv(image, lower, upper) -> tuple[np.ndarray, np.ndarray]  # (mask, result)
def suggest_range(image, x, y, radius=5, tolerance=(10, 60, 60)) -> tuple[lower, upper]
def clean_mask(mask, kernel_size, min_area=0) -> np.ndarray
def region_stats(mask) -> dict   # {"coverage": float, "n_regions": int, "areas": list}
```
- Màu đỏ vòng (H ở hai đầu): nếu lower[0] > upper[0] thì hiểu là vòng qua 179 -> 0.
- Tham số sai (H ngoài 0-179, S/V ngoài 0-255, min > max) thì ném `ValueError`.
- hsv_core.py còn có hàm lớp dưới (`apply_hsv_segmentation`, `make_overlay`, `mask_statistics`...), `segment_hsv` là cửa vào chính cho GUI.
- Xử lý batch và video nằm ở `hsv_media.py`.
## TV2 - gui/
- Gọi các hàm trên, không tự cài đặt thuật toán.
- Dùng `hsv_core_mock.py` cho đến khi TV1 xong, sau đó đổi dòng import.

### TV3 - data/, tests/, tools/
- Ảnh đặt trong `data/images/`, mask chuẩn trong `data/masks/` (cùng tên file, PNG, giá trị 0/255).
- `seg_metrics.py`: `iou(pred, gt)`, `dice(pred, gt)` cho hai mask nhị phân cùng kích thước.
- `presets.json`: giá trị preset đã tinh chỉnh từ ảnh thật (GUI đọc file này).
- Chạy test: `python -m unittest discover -s tests -t . -v`
- Chạy đánh giá: `python tools/run_eval.py`