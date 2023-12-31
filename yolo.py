# import cv2
import torch
from pathlib import Path

# 加载YOLOv5模型


class Yolo:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    def detect(self, frame):
        i = 0
        array = []
        results = self.model(frame)

        # 获取检测结果
        predictions = results.pred[0]

        for pred in predictions:
            class_id, confidence, bbox = pred[5], pred[4], pred[:4]
            class_name = self.model.names[int(class_id)]

            if confidence > 0.5 and class_name == "person":  # 可以自定义阈值
                x = int(bbox[0])
                y = int(bbox[1])
                ix = int(bbox[2])
                iy = int(bbox[3])
                if (abs(x - ix) > 10 and abs(y - iy) > 10):
                    w, h = abs(x - ix), abs(y - iy)  # w 宽度 h 高度
                    ix, iy = min(x, ix), min(y, iy)  # 更新起点为左上角
                    array.append([w, h, ix, iy, class_name])
        return array if array is not None else None
