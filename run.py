import cv2
from time import time

import kcftracker
import yolo

selectingObject = False  # 全局变量，是否正在选择对象
initTracking = True  # 全局变量，是否正在初始化追踪
onTracking = False  # 全局变量，是否正在追踪
useYolo = False

# 若有必要，此处可优化重构，使用全局状态存储，方便处理复杂回调事件。
# 每次状态更新理应更新全部三个变量，但作者编码时偷懒了。实际上三个状态不会重叠，只需要一个变量就行了。

ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1  # 回调函数检测间隔时间
duration = 0.01  # 持续时间，此处影响初始帧率


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
    # config hog, fixed_window, multiscale
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    tracker = kcftracker.KCFTracker(False, True, True)
    detecter = yolo.Yolo

    cv2.namedWindow('tracking')

    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv.VideoCapture.read() -> retval, image
        # retval:返回值,bool
        # image:array,图像RGB

        if not ret:
            break

        if (useYolo):
            pass
        elif (initTracking):
            Object = detecter.detect(frame)
            ix, iy, w, h, class_name = Object
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            # print([ix, iy, w, h])
            tracker.init([ix, iy, w, h], frame)  # 追踪器初始化，输入当前框与当前帧
            initTracking = False
            onTracking = True
        elif (onTracking):
            t0 = time()
            boundingbox = tracker.update(frame)
            t1 = time()  # 记录追踪器计算时间
            boundingbox = list(map(int, boundingbox))  # 数据类型的转换，理解时可以忽略
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (
                boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)  # 更新追踪盒位置
            duration = 0.8 * duration + 0.2 * \
                (t1 - t0)  # 使用平滑帧率代替真实帧率，让用户能更直观地评估当前性能

        image = frame
        res = tracker.res
        fuse_img = matrix_insert(image, res)
        _, pv, _, _ = cv2.minMaxLoc(res)

        # 帧率显示
        cv2.putText(frame, 'peak value:' + str(pv) + 'FPS: ' + str(1 / duration)
                    [:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', fuse_img if fuse_img is not None else frame)  # 更新帧
        c = cv2.waitKey(inteval) & 0xFF  # 将获取到的值赋给c,使用位掩码确保值在0到255之间
        if c == 27 or c == ord('q'):
            break
        # 检测到按下q则退出循环
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口
