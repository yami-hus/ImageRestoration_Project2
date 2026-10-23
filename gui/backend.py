"""Điểm chuyển đổi duy nhất giữa bản giả (mock) và bản thật của TV1.
Khi TV1 merge xong hsv_core.py: đổi USE_MOCK = False là xong."""
USE_MOCK = True

if USE_MOCK:
    from hsv_core_mock import segment_hsv, suggest_range, clean_mask, region_stats
else:
    from hsv_core import segment_hsv, suggest_range, clean_mask, region_stats

__all__ = ["segment_hsv", "suggest_range", "clean_mask", "region_stats"]
