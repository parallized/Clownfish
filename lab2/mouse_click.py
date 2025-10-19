import win32gui
import win32api
import win32con
import time

def click_window(window_title, x, y):
    """
    在指定窗口的指定位置点击鼠标左键

    :param window_title: 窗口标题
    :param x: 点击位置的横坐标（相对窗口左上角）
    :param y: 点击位置的纵坐标（相对窗口左上角）
    :param delay: 点击按下和抬起间隔，单位秒
    """
    # 找到窗口句柄
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        print(f"未找到窗口: {window_title}")
        return False

    # 获取窗口左上角屏幕坐标
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    screen_x = left + x
    screen_y = top + y

    # 移动鼠标到指定位置
    win32api.SetCursorPos((screen_x, screen_y))
    time.sleep(0.01)

    # 鼠标左键按下
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    # 鼠标左键抬起
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    return True
