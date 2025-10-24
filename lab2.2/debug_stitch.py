import cv2
import numpy as np
from match import match_and_transform, stitch_map

def test_stitch():
    """测试拼接功能"""
    # 创建两个测试图像
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 在img1中画一个矩形
    cv2.rectangle(img1, (50, 50), (100, 100), (0, 255, 0), -1)
    
    # 在img2中画一个稍微偏移的矩形
    cv2.rectangle(img2, (60, 60), (110, 110), (255, 0, 0), -1)
    
    print("测试图像创建完成")
    
    # 尝试匹配和拼接
    H = match_and_transform(img1, img2)
    
    if H is not None:
        print("找到变换矩阵，开始拼接...")
        result = stitch_map(img1, img2, H)
        
        # 显示结果
        cv2.imshow("原始图像1", img1)
        cv2.imshow("原始图像2", img2)
        cv2.imshow("拼接结果", result)
        
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未能找到变换矩阵")

if __name__ == "__main__":
    test_stitch()
