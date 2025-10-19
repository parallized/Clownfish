from ultralytics import YOLO
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, cv_img):
        # 推理：传入 OpenCV 格式的图像 (BGR)
        results = self.model.predict(
            source=cv_img,
            conf=self.conf,
            verbose=False
        )
        # 可视化结果（框框 + 标签）
        annotated_frame = results[0].plot()
        return annotated_frame, results
