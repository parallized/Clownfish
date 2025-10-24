import cv2
import numpy as np

class MapMatcher2D:
    def __init__(self):
        # 针对2D地图优化的特征检测器
        self.orb = cv2.ORB_create(
            nfeatures=1000,      # 减少特征点数量，提高速度
            scaleFactor=1.1,     # 更小的尺度因子，适合2D地图
            nlevels=6,           # 减少金字塔层数
            edgeThreshold=15,    # 边缘阈值
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # 简化的匹配器
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 用于模板匹配的窗口大小
        self.template_size = (50, 50)
        
    def detect_features_2d(self, image):
        """针对2D地图优化的特征检测"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 增强对比度，提高特征检测效果
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 检测ORB特征点
        kp, des = self.orb.detectAndCompute(enhanced, None)
        
        return kp, des, enhanced
    
    def match_features_2d(self, des1, des2):
        """2D地图特征匹配"""
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return []
        
        try:
            # 使用BF匹配器
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 过滤好的匹配点
            good_matches = []
            for match in matches:
                if match.distance < 40:  # 降低距离阈值，提高精度
                    good_matches.append(match)
            
            return good_matches
        except:
            return []
    
    def find_affine_transform(self, kp1, kp2, matches):
        """计算仿射变换矩阵（适合2D平面地图）"""
        if len(matches) < 3:
            print(f"匹配点不足，需要至少3个，当前只有{len(matches)}个")
            return None
        
        # 获取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        print(f"匹配点坐标 - 大地图: {len(pts1)} 个点, 新补丁: {len(pts2)} 个点")
        
        try:
            # 计算仿射变换矩阵（比透视变换更适合2D地图）
            M, mask = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, 
                                                ransacReprojThreshold=3.0, maxIters=2000)
            
            print(f"RANSAC计算得到的变换矩阵: \n{M}")
            
            if M is not None:
                # 检查变换的合理性
                if self.is_valid_affine_transform(M):
                    print("变换矩阵验证通过")
                    return M
                else:
                    print("变换矩阵验证失败")
            else:
                print("RANSAC计算失败")
        except Exception as e:
            print(f"计算仿射变换时出错: {e}")
        
        return None
    
    def is_valid_affine_transform(self, M):
        """检查仿射变换矩阵是否合理"""
        # 检查缩放因子
        scale_x = np.sqrt(M[0,0]**2 + M[0,1]**2)
        scale_y = np.sqrt(M[1,0]**2 + M[1,1]**2)
        
        # 限制缩放范围
        if scale_x < 0.8 or scale_x > 1.2 or scale_y < 0.8 or scale_y > 1.2:
            return False
        
        # 检查旋转角度
        angle = np.arctan2(M[0,1], M[0,0])
        if abs(angle) > np.pi/6:  # 限制在30度以内
            return False
        
        # 检查平移量
        tx, ty = M[0,2], M[1,2]
        if abs(tx) > 100 or abs(ty) > 100:  # 限制平移距离
            return False
        
        return True
    
    def template_match_fallback(self, img1, img2):
        """模板匹配作为备选方案"""
        print("尝试模板匹配...")
        
        # 转换为灰度图
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        
        print(f"模板匹配 - 大地图: {w1}x{h1}, 新补丁: {w2}x{h2}")
        
        if h1 > 50 and w1 > 50 and h2 > 50 and w2 > 50:
            # 提取中心区域作为模板
            template_size = min(50, h1//3, w1//3)
            template = gray1[h1//2-template_size//2:h1//2+template_size//2, 
                           w1//2-template_size//2:w1//2+template_size//2]
            
            print(f"模板尺寸: {template.shape}")
            
            # 模板匹配
            result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            print(f"模板匹配结果: 最大相似度 = {max_val:.3f}")
            
            if max_val > 0.4:  # 降低匹配阈值
                # 计算偏移量
                tx = max_loc[0] - (w1//2-template_size//2)
                ty = max_loc[1] - (h1//2-template_size//2)
                
                print(f"计算得到的偏移量: tx={tx}, ty={ty}")
                
                # 创建仿射变换矩阵
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                return M
            else:
                print("模板匹配相似度过低")
        else:
            print("图像尺寸太小，无法进行模板匹配")
        
        return None

# 创建全局匹配器实例
matcher = MapMatcher2D()

def stitch_map_2d(big_map, new_patch, M):
    """2D地图拼接，使用仿射变换"""
    h, w = big_map.shape[:2]
    
    print(f"拼接前大地图尺寸: {big_map.shape}, 新补丁尺寸: {new_patch.shape}")
    print(f"变换矩阵: \n{M}")
    
    # 使用仿射变换
    result = cv2.warpAffine(new_patch, M, (w, h))
    
    # 创建掩码，只更新非零像素
    mask = np.sum(result, axis=2) > 0
    mask_count = np.sum(mask)
    print(f"有效像素数量: {mask_count}")
    
    if mask_count > 0:
        # 混合图像（使用加权平均，降低新图像权重以减少累积误差）
        alpha = 0.6  # 降低新图像权重
        big_map[mask] = (1 - alpha) * big_map[mask] + alpha * result[mask]
        print(f"成功拼接，更新了 {mask_count} 个像素")
    else:
        print("警告：没有有效像素被更新")
    
    return big_map

def match_and_transform(old_map, new_patch):
    """2D地图匹配并计算变换矩阵"""
    # 检测特征点
    kp1, des1, enhanced1 = matcher.detect_features_2d(old_map)
    kp2, des2, enhanced2 = matcher.detect_features_2d(new_patch)
    
    if kp1 is None or kp2 is None or des1 is None or des2 is None:
        print("特征检测失败")
        return None
    
    # 匹配特征点
    matches = matcher.match_features_2d(des1, des2)
    
    print(f"检测到 {len(kp1)} 和 {len(kp2)} 个特征点，匹配到 {len(matches)} 个")
    
    # 尝试计算仿射变换矩阵
    M = None
    if len(matches) >= 3:
        M = matcher.find_affine_transform(kp1, kp2, matches)
    
    # 如果特征匹配失败，尝试模板匹配
    if M is None:
        print("特征匹配失败，尝试模板匹配...")
        M = matcher.template_match_fallback(old_map, new_patch)
    
    if M is not None:
        print("找到有效的变换矩阵")
        return M
    else:
        print("所有匹配方法都失败")
        return None

# 为了兼容性，保留原函数名
def stitch_map(big_map, new_patch, H):
    """兼容性函数，将仿射变换矩阵转换为透视变换矩阵"""
    if H is None:
        return big_map
    
    # 将2x3仿射变换矩阵转换为3x3透视变换矩阵
    if H.shape == (2, 3):
        H_3x3 = np.vstack([H, [0, 0, 1]])
        return stitch_map_2d(big_map, new_patch, H)
    else:
        # 如果已经是3x3矩阵，使用透视变换
        h, w = big_map.shape[:2]
        result = cv2.warpPerspective(new_patch, H, (w, h))
        mask = np.sum(result, axis=2) > 0
        alpha = 0.6
        big_map[mask] = (1 - alpha) * big_map[mask] + alpha * result[mask]
        return big_map
