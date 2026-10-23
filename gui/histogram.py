"""TV2: histogram kênh H (TODO: nhúng matplotlib FigureCanvasQTAgg vào GUI)."""
import cv2


def compute_h_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
