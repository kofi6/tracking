import cv2
from time import time
import yolo
import criterion
import modelMannger

# [config]
useYolo = True
status = 0
# 0 selectingObject
# 1 initTracking
# 2 onTracking
# 3 pause

# parameter init
res = None
judge = True
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
inteval = 1  # 回调函数检测间隔时间
duration = 0.01  # 持续时间，此处影响初始帧率


def draw_boundingbox(event, x, y, p, q):
    global status, ix, iy, cx, cy, w, h

    # 按下左键，设置状态未正在选择目标，并取消追踪状态

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        cx, cy = x, y
        status == 0

    # 鼠标开始移动

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    # 鼠标左键抬起，取消正在选择目标状态，判断是否大于10*10

    elif event == cv2.EVENT_LBUTTONUP:
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)  # w 宽度 h 高度
            ix, iy = min(x, ix), min(y, iy)  # 更新起点为左上角
            status = 1

    elif event == cv2.EVENT_RBUTTONDOWN:
        pass

        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            status = 1


def matrix_insert(src1, src2):
    if src2 is not None:
        row, col = src1.shape[0]-src2.shape[0], 0
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 0] = src2
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 1] = src2
        src1[row:row+src2.shape[0], col:col+src2.shape[1], 2] = src2

        return src1
    else:
        return None


if __name__ == '__main__':
    cv2.namedWindow('tracking')
    cap = cv2.VideoCapture(0)
    tracker = modelMannger.modelA()
    modelC = modelMannger.modelC()

    if useYolo:
        detecter = yolo.Yolo()
    else:
        cv2.setMouseCallback('tracking', draw_boundingbox)

    while (cap.isOpened()):

        if res is not None:
            judge = criterion.Criterion(res)
            if not judge:
                print("目标丢失")
                w, h, ix, iy, class_name = modelC.reSelect(
                    tracker, detecter, frame)
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h),
                              (0, 255, 255), 2)
                tracker.init([ix, iy, w, h], frame)
        ret, frame = cap.read()

        if not ret:
            break
        if status == 0:  # draw box
            if useYolo:
                Object = detecter.detect(frame)
                w, h, ix, iy, class_name = Object[0]
                modelC.init([ix, iy, w, h], frame)
                status = 1  # 如果没有开启yolo，则状态转换在drawbox中进行。不写进来是因为左键按下时需要等待
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif status == 1:  # initracking
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            tracker.init([ix, iy, w, h], frame)  # 追踪器初始化，输入当前框与当前帧
            status = 2
        elif status == 2:  # ontracking
            t0 = time()
            boundingbox = tracker.update(frame, judge)
            t1 = time()
            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (
                boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)  # 更新追踪盒位置
            duration = 0.8 * duration + 0.2 * \
                (t1 - t0)  # 使用平滑帧率代替真实帧率，让用户能更直观地评估当前性能

        image = frame
        res = tracker.res
        fuse_img = matrix_insert(image, res)

        # 信息显示
        if res is not None:
            cv2.putText(frame, 'APCE:'+str(criterion.APCE(res))+'FPS: ' + str(1 / duration)
                        [:4].strip('.'), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'FPS: ' + str(1 / duration)
                        [:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', fuse_img if fuse_img is not None else frame)  # 更新帧
        c = cv2.waitKey(inteval) & 0xFF  # 将获取到的值赋给c,使用位掩码确保值在0到255之间
        if c == 27 or c == ord('q'):
            break
        # 检测到按下q则退出循环
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口
