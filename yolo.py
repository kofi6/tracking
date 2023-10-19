#import cv2
import torch
from pathlib import Path

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# 打开摄像头
#cap = cv2.VideoCapture(0)  # 0表示默认摄像头
class Yolo:
    def __init__(self,frame=None) -> None:
        assert frame,"图像传入错误"

    def detect(frame):
        while True:
            #ret, frame = cap.read()
            #if not ret:
            #   break
            #frame="test.jpg"
            # 使用YOLOv5模型进行目标检测
            results = model(frame)

            # 获取检测结果
            predictions = results.pred[0]

            for pred in predictions:
                class_id, confidence, bbox = pred[5], pred[4], pred[:4]
                class_name = model.names[int(class_id)]

                if confidence > 0.5 and class_name=="person":  # 可以自定义阈值
                    # 绘制边界框
                    #pt1 = (int(bbox[0]), int(bbox[1]))
                    #pt2 = (int(bbox[2]), int(bbox[3]))
                    x=int(bbox[0])
                    y=int(bbox[1])
                    ix=int(bbox[2])
                    iy=int(bbox[3])
                    if (abs(x - ix) > 10 and abs(y - iy) > 10):
                        w, h = abs(x - ix), abs(y - iy)  # w 宽度 h 高度
                        ix, iy = min(x, ix), min(y, iy)  # 更新起点为左上角
                    return ix,iy,w,h,class_name
                else:
                    return None
            #cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            #cv2.putText(frame, f'{class_name} {confidence:.2f}', (pt1[0], pt1[1] - 5),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    #cv2.imshow('YOLOv5 Object Detection', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

# 释放摄像头和关闭窗口
#cap.release()
#cv2.destroyAllWindows()
