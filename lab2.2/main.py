import win32gui
import cv2
from capture import capture_window, clip_img
from match import stitch_map, match_and_transform
import numpy as np
import time

class MapStitcher:
    def __init__(self, window_title="魔兽世界", max_size=1200):
        """
        初始化地图拼接器
        :param window_title: 游戏窗口标题
        :param max_size: 大地图最大尺寸
        """
        self.max_size = max_size
        self.big_map = None
        self.big_map_center = None
        self.is_first_frame = True
        
        # 查找游戏窗口
        self.hwnd = win32gui.FindWindow(None, window_title)
        if not self.hwnd:
            raise Exception(f"找不到窗口: {window_title}")
        
        print(f"找到游戏窗口: {window_title}")
    
    def get_minimap_image(self):
        """
        获取游戏右上角小地图图像
        """
        try:
            # 截取游戏窗口右上角小地图区域
            # 根据你的需求调整坐标和尺寸
            pil_img = clip_img(capture_window(self.hwnd), 20, 3270, 140, 140)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"获取小地图图像失败: {e}")
            return None
    
    def initialize_big_map(self, first_minimap):
        """
        初始化大地图
        """
        h, w = first_minimap.shape[:2]
        print(f"第一帧小地图尺寸: {w}x{h}")
        
        # 创建1200x1200的大地图
        self.big_map = np.zeros((self.max_size, self.max_size, 3), dtype=np.uint8)
        
        # 将第一张小地图放在大地图中央
        center_x = self.max_size // 2
        center_y = self.max_size // 2
        start_x = center_x - w // 2
        start_y = center_y - h // 2
        
        print(f"将小地图放置在大地图位置: ({start_x}, {start_y})")
        
        # 确保不会越界
        end_x = min(start_x + w, self.max_size)
        end_y = min(start_y + h, self.max_size)
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        self.big_map[start_y:end_y, start_x:end_x] = first_minimap[:actual_h, :actual_w]
        self.big_map_center = (center_x, center_y)
        
        print(f"初始化大地图完成，尺寸: {self.max_size}x{self.max_size}")
        print(f"实际放置区域: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        return True
    
    def stitch_minimap_to_bigmap(self, new_minimap):
        """
        将新的小地图拼接到大地图上
        """
        if self.big_map is None:
            return False
        
        try:
            # 使用特征匹配找到变换矩阵
            H = match_and_transform(self.big_map, new_minimap)
            if H is not None:
                # 拼接图像
                self.big_map = stitch_map(self.big_map, new_minimap, H)
                return True
            else:
                print("特征匹配失败，跳过此帧")
                return False
        except Exception as e:
            print(f"图像拼接失败: {e}")
            return False
    
    def resize_big_map_if_needed(self):
        """
        如果大地图过大，进行裁剪
        """
        if self.big_map is None:
            return
        
        h, w = self.big_map.shape[:2]
        if h > self.max_size or w > self.max_size:
            # 裁剪到最大尺寸
            start_y = max(0, (h - self.max_size) // 2)
            start_x = max(0, (w - self.max_size) // 2)
            
            self.big_map = self.big_map[
                start_y:start_y+self.max_size,
                start_x:start_x+self.max_size
            ]
            print(f"大地图已裁剪到: {self.big_map.shape}")
    
    def run(self):
        """
        运行地图拼接主循环
        """
        print("开始地图拼接...")
        print("按 'q' 键退出")
        
        frame_count = 0
        success_count = 0
        
        while True:
            # 获取当前小地图图像
            minimap = self.get_minimap_image()
            if minimap is None:
                continue
            
            frame_count += 1
            
            if self.is_first_frame:
                # 第一帧，初始化大地图
                if self.initialize_big_map(minimap):
                    self.is_first_frame = False
                    success_count += 1
            else:
                # 后续帧，拼接到大地图
                if self.stitch_minimap_to_bigmap(minimap):
                    success_count += 1
                
                # 定期检查并调整大地图大小
                if frame_count % 10 == 0:
                    self.resize_big_map_if_needed()
            
            # 显示大地图
            if self.big_map is not None:
                # 缩放显示图像以适应屏幕
                display_map = cv2.resize(self.big_map, (800, 800))
                cv2.imshow("World Map", display_map)
                
                # 显示统计信息
                info_text = f"Frames: {frame_count}, Success: {success_count}"
                cv2.putText(display_map, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("World Map", display_map)
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # 按 'r' 重置大地图
                print("重置大地图...")
                self.is_first_frame = True
                self.big_map = None
        
        cv2.destroyAllWindows()
        print(f"地图拼接完成！总共处理 {frame_count} 帧，成功拼接 {success_count} 帧")

def main():
    try:
        stitcher = MapStitcher()
        stitcher.run()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()