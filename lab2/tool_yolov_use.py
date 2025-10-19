import time
import cv2
import numpy as np
from capture import capture_window_bottom_80
from detector import YOLOv8Detector
import win32gui
from mouse_click import click_window

# 初始化
window_title = "Plants vs. Zombies"
hwnd = win32gui.FindWindow(None, window_title)
detector = YOLOv8Detector("best.pt", conf=0.1) # 模型（yolov8 自带的测试用模型，可以识别常见物体）

start_time = time.time()

while True:
    # 截取窗口图像
    pil_img = capture_window_bottom_80(hwnd)

    # YOLO 检测
    annotated, results = detector.detect(pil_img)

    # 显示结果
    cv2.imshow("YOLOv8 Real-Time", annotated)
    boxes = results[0].boxes # results 是 detector.detect 返回的结果，是一个列表
    # boxes 是 results.boxes
    # results 是 detector.detect 返回的结果
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])       # 类别索引
        score = box.conf[0].item()     # 置信度
        cls_name = results[0].names[cls_id]  # 映射成名字

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        print(f"目标 {cls_name} 坐标中心: ({cx},{cy}), 置信度: {score}")

        # 如果要自动点击
        if cls_name == "Sun":q
            click_window("Plants vs. Zombies", cx, cy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
