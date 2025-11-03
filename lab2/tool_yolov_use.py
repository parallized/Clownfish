import time
import cv2
import numpy as np
from capture import capture_window_bottom_80
from detector import YOLOv8Detector
import win32gui
from mouse_click import click_window

# Sun Resolution
def getAllSunPosition(results):
    sunPositions = []

    boxes = results[0].boxes # results 是 detector.detect 返回的结果，是一个列表
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist() # 解构
        cls_id = int(box.cls[0])       # 类别索引
        score = box.conf[0].item()     # 置信度
        cls_name = results[0].names[cls_id]  # 映射成名字
        if cls_name == "Sun":
            sunPositions.append((x1, y1, x2, y2))
    return sunPositions

# 初始化
window_title = "Plants vs. Zombies"
hwnd = win32gui.FindWindow(None, window_title) # 返回 hwnd 句柄
detector = YOLOv8Detector("best.pt", conf=0.1) # 模型（yolov8 自带的测试用模型，可以识别常见物体）

start_time = time.time()

while True:
    # 截取窗口图像
    pil_img = capture_window_bottom_80(hwnd)

    # YOLO 检测
    annotated, results = detector.detect(pil_img) # 进行一次图形识别

    # 显示结果
    cv2.imshow("YOLOv8 Real-Time", annotated)

    # 获取所有阳光位置
    suns = getAllSunPosition(results)

    # 点击所有阳光位置
    for sun in suns:
        click_window("Plants vs. Zombies", sun[0], sun[1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
