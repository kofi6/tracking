import cv2
from time import time

import kcftracker
import yolo


def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    # 按下左键，设置状态未正在选择目标，并取消追踪状态

    if event == cv2.EVENT_LBUTTONDOWN:
        status == 1
        ix, iy = x, y
        cx, cy = x, y

    # 鼠标开始移动

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    # 鼠标左键抬起，取消正在选择目标状态，判断是否大于10*10

    elif event == cv2.EVENT_LBUTTONUP:
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)  # w 宽度 h 高度
            ix, iy = min(x, ix), min(y, iy)  # 更新起点为左上角
            status = 2

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False

        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True


ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
inteval = 1  # 回调函数检测间隔时间
duration = 0.01  # 持续时间，此处影响初始帧率

status = 0  # 初始化为0时不进行yolo检测，初始化为1时进行yolo检测
# 0 selectingObject
# 1 initTracking
# 2 onTracking
# 3 pause


def matrix_insert(src1, src2):
    if src2 is not None:
        row, col = src1.shape[0]-src2.shape[0], 0
        # roi = src2[y:y+height, x:x+width]  # 指定感兴趣区域
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 0] = src2
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 1] = src2
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 2] = src2

        return src1
    else:
        return None


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('tracking')
    # config hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    tracker = kcftracker.KCFTracker(False, True, True)

    if status == 1:
        detecter = yolo.Yolo
    elif status == 0:
        cv2.setMouseCallback('tracking', draw_boundingbox)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv.VideoCapture.read() -> retval, image
        # retval:返回值,bool
        # image:array,图像RGB

        if not ret:
            break
        if status == 0:
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif status == 1:
            if detecter is not None:
                Object = detecter.detect(frame)
                ix, iy, w, h, class_name = Object
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            tracker.init([ix, iy, w, h], frame)  # 追踪器初始化，输入当前框与当前帧
            status = 2
        elif status == 2:
            t0 = time()
            boundingbox = tracker.update(frame)
            t1 = time()  # 记录追踪器计算时间
            boundingbox = list(map(int, boundingbox))  # 数据类型的转换，理解时可以忽略
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (
                boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)  # 更新追踪盒位置
            duration = 0.8 * duration + 0.2 * \
                (t1 - t0)  # 使用平滑帧率代替真实帧率，让用户能更直观地评估当前性能
        elif status == 3:
            break  # 如果状态为3，则将挂起

        image = frame
        res = tracker.res
        fuse_img = matrix_insert(image, res)

        # 帧率显示
        cv2.putText(frame, 'FPS: ' + str(1 / duration)
                    [:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('tracking', fuse_img if fuse_img is not None else frame)  # 更新帧
        c = cv2.waitKey(inteval) & 0xFF  # 将获取到的值赋给c,使用位掩码确保值在0到255之间
        if c == 27 or c == ord('q'):
            break
        # 检测到按下q则退出循环
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口
