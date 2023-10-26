# 本文件创造了一个模型队列，辅助主KCF模型训练
import kcftracker
import criterion

judge = criterion.Criterion
# raw model


def overlap(pt1, pt2, pt3, pt4):
    # 计算A框和B框的左上角和右下角坐标
    ax1, ay1 = pt1
    ax2, ay2 = pt2
    bx1, by1 = pt3
    bx2, by2 = pt4

    # 计算重叠区域的左上角和右下角坐标
    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)

    # 计算重叠区域的宽度和高度
    width = max(0, overlap_x2 - overlap_x1)
    height = max(0, overlap_y2 - overlap_y1)

    # 计算重叠面积
    overlap_area = width * height
    overlap_area = overlap_area/(abs(ax1-ax2)*abs(ay1-ay2))

    # 如果没有重叠，重叠面积为0
    if width <= 0 or height <= 0:
        overlap_area = 0

    return overlap_area


class modelA(kcftracker.KCFTracker):
    def __init__(self):
        super().__init__()


# High quality model array
class modelB():
    def __init__(self):
        super.__init__()
        self.array = []
        self.res = None

    def insertArray(self, KCF):
        assert type(KCF).__name__ == "kcftracker"
        mask = 20  # 存储的模型数量
        self.array = self.array.insert((len(self.array)+1) & mask, KCF)

    def resp(self, frame):
        for model in self.array:
            model.update(frame, False)  # 不训练
            self.res = self.res+model.res
        self.res = self.res/len(self.array)


# model trained by YOLO detect frame
class modelC(kcftracker.KCFTracker):
    def __init__(self):
        super().__init__()

    def reSelect(self, kcf, yolo, frame):
        objects = yolo.detect(frame)
        bigger_area = 0
        for obj in objects:
            bbox = kcf.update(frame, False)
            bbox = list(map(int, bbox))
            pt1, pt2 = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            pt3 = obj[2], obj[3]
            pt4 = obj[0]+obj[2], obj[1]+obj[3]
            area = overlap(pt1, pt2, pt3, pt4)
            if area > bigger_area:
                bigger_area = area
                self.best_roi = obj

        return self.best_roi
