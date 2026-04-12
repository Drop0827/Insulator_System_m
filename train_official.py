import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载官方模型 (推荐先从 yolov11n 或 yolov11s 开始)
    # 直接加载 .pt 文件，会自动下载权重并根据权重构建模型结构
    model = YOLO('yolo11n.pt')

    # 2. 开始训练
    model.train(
        data=r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\data.yaml",
        task='detect',
        epochs=200,
        imgsz=640,
        batch=32,  # <--- 调大 Batch，显存够就用32或64
        workers=4,  # <--- 增加线程，加快图片预处理速度
        cache=False,  # <--- 开启缓存，减少磁盘IO压力
        device='0',
        optimizer='auto',
        amp=True,  # 3060 支持混合精度，必须开启
        project='runs/train',
        name='v11n_baseline',
    )