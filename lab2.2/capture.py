import win32gui
import win32ui
import win32con
from PIL import Image
import mss

def clip_img(img, top, left, width, height):
    return img.crop((left, top, left + width, top + height))

def capture_window(hwnd):
    # 获取窗口矩形
    rect = win32gui.GetWindowRect(hwnd)
    x1, y1, x2, y2 = rect
    width, height = x2 - x1, y2 - y1

    top = y1
    mon = {"top": top, "left": x1, "width": width, "height": height}
    
    with mss.mss() as sct:
        img = sct.grab(mon)
        return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")