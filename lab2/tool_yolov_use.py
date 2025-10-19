import time
import cv2
import numpy as np
from capture import capture_window_bottom_80
from detector import YOLOv8Detector
import win32gui

# 初始化
window_title = "魔兽世界"
hwnd = win32gui.FindWindow(None, window_title)
detector = YOLOv8Detector("yolov8n.pt", conf=0.45) # 模型（yolov8 自带的测试用模型，可以识别常见物体）

start_time = time.time()

while True:
    # 截取窗口图像
    pil_img = capture_window_bottom_80(hwnd)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # YOLO 检测
    annotated, results = detector.detect(cv_img)

    # 显示结果
    cv2.imshow("YOLOv8 Real-Time", annotated)

    # FPS 计算
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.1f}")
    start_time = end_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
