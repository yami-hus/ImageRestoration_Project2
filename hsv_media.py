import os

import cv2
import numpy as np

from hsv_core import HSVParams, validate_hsv_params, apply_hsv_segmentation, make_overlay

VALID_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def _read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path, image):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True


# %% [XỬ LÝ HÀNG LOẠT ẢNH]
def process_hsv_batch_images(input_dir, output_dir, params, save_mask=True):
    validate_hsv_params(params)
    if not input_dir or not os.path.isdir(input_dir):
        raise ValueError(f"Thư mục đầu vào không tồn tại: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    skipped = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(VALID_IMAGE_EXT):
            continue
        stem = os.path.splitext(filename)[0]
        try:
            img = _read_image(os.path.join(input_dir, filename))
            if img is None:
                skipped.append((filename, "không đọc được ảnh (file hỏng hoặc sai định dạng)"))
                continue
            mask, result = apply_hsv_segmentation(img, params)
            if not _write_image(os.path.join(output_dir, f"result_{stem}.png"), result):
                skipped.append((filename, "không ghi được ảnh kết quả"))
                continue
            if save_mask:
                _write_image(os.path.join(output_dir, f"mask_{stem}.png"), mask)
            processed += 1
        except OSError as e:
            skipped.append((filename, f"lỗi file: {e}"))
    return {"processed": processed, "skipped": skipped}


# %% [THÔNG TIN VIDEO]
def get_hsv_video_info(video_path):
    if not video_path or not isinstance(video_path, str) or not os.path.isfile(video_path):
        raise ValueError(f"Đường dẫn video không hợp lệ: {video_path}")
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        return {
            "fps": fps if fps and fps > 0 else 25.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


# %% [GENERATOR VIDEO]
def _resolve_params(params):
    p = params() if callable(params) else params
    validate_hsv_params(p)
    return p


def process_hsv_video_generator(video_path, params):
    if not video_path or not isinstance(video_path, str) or not os.path.isfile(video_path):
        raise ValueError(f"Đường dẫn video không hợp lệ: {video_path}")
    _resolve_params(params)

    def _frames():
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Không thể mở video: {video_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                mask, result = apply_hsv_segmentation(frame, _resolve_params(params))
                yield frame, mask, result
        finally:
            cap.release()

    return _frames()


# %% [LƯU VIDEO KẾT QUẢ]
def save_hsv_video(input_path, output_path, params, output="result", overlay_alpha=0.5):

    if output not in ("result", "overlay", "mask"):
        raise ValueError(f"output phải là 'result', 'overlay' hoặc 'mask', nhận được {output!r}")
    validate_hsv_params(params)
    info = get_hsv_video_info(input_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             info["fps"], (info["width"], info["height"]))
    if not writer.isOpened():
        raise ValueError(f"Không thể tạo file video đầu ra: {output_path}")

    written = 0
    try:
        for frame, mask, result in process_hsv_video_generator(input_path, params):
            if output == "result":
                out_frame = result
            elif output == "overlay":
                out_frame = make_overlay(frame, mask, alpha=overlay_alpha)
            else:
                out_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            writer.write(out_frame)
            written += 1
    finally:
        writer.release()
    return written


# %% [THỬ NHANH - CHẠY ĐỘC LẬP]
if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) < 3:
        raise SystemExit("Cách dùng: python hsv_media.py <thư_mục_ảnh_vào> <thư_mục_ra>")
    p = HSVParams(h_min=35, h_max=85, s_min=60, v_min=60)
    report = process_hsv_batch_images(sys.argv[1], sys.argv[2], p)
    print(f"Đã xử lý {report['processed']} ảnh")
    for name, reason in report["skipped"]:
        print(f"  Bỏ qua {name}: {reason}")