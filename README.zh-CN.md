# 简介
本项目创建于大连海事大学2024届电子信息课程设计。课题内容为目标追踪。使用了KCF作为目标追踪的核心算法，使用yolov5对初始帧进行分类并定位。
# 环境
anaconda 3，其余环境在yolo_py39.yml当中
# KCF
KCF算法来自于https://github.com/LCorleone/KCF_py3
略有修改

KCF原作者https://www.robots.ox.ac.uk/~joao/#research

# yolo
使用yolov5m模型作为分类模型
yolo：https://github.com/ultralytics/yolov5

# 问题
yolo和HOG不能同时开启，正在修复

已定位问题：fhog开发时依赖numpy库与最新的numpy不兼容，具体来说，是numpy.int类型。
考虑：重构fhog.py或启动另一个conda环境，将fhog放入其中运行。但解决优先级很低，定性来看，不开启HOG依然可以获得较为不错的追踪效果。    


