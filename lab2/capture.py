import win32gui
import win32ui
import win32con
from PIL import Image
import mss

def capture_window_bottom_80(hwnd):
    # 获取窗口矩形
    rect = win32gui.GetWindowRect(hwnd)
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1
    
    # 截取底部 80%
    top = y1 + int(height * 0.2)
    mon = {"top": top, "left": x1, "width": width, "height": int(height * 0.8)}
    
    with mss.mss() as sct:
        img = sct.grab(mon)
        return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")