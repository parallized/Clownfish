import win32gui

def enum_windows_callback(hwnd, titles):
    title = win32gui.GetWindowText(hwnd)
    if win32gui.IsWindowVisible(hwnd) and title.strip():
        titles.append(title)

def list_all_window_titles():
    titles = []
    win32gui.EnumWindows(enum_windows_callback, titles)
    return titles

if __name__ == "__main__":
    all_titles = list_all_window_titles()
    
    # 保存到文件
    with open("window_title.log", "w", encoding="utf-8") as f:
        for t in all_titles:
            f.write(t + "\n")
    
    print(f"已保存 {len(all_titles)} 个窗口标题到 window_title.log")
