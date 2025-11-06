import win32gui
import win32ui
import win32con
import time
from PIL import Image
import numpy as np
import cv2
from capture import capture_window_bottom_80

window_title = "魔兽世界"
hwnd = win32gui.FindWindow(None, window_title)
if not hwnd:
    raise Exception(f"找不到窗口: {window_title}") # throw error 抛出异常 终止程序

startTime = time.time()

while True:
    # 截取
    pil_img = capture_window_bottom_80(hwnd) # 截图
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # 转换为 OpenCV 格式    
    cv2.imshow("Window Capture (bottom 80%)", cv_img) # 显示截图

    # 计算 FPS
    endTime = time.time()
    fps = 1 / (endTime - startTime) # 计算 FPS
    print(f"FPS: {fps:.1f}")
    startTime = endTime

    # 按下 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
