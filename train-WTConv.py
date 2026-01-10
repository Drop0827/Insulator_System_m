import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载改进后的网络结构
    model = YOLO('yolo11-C3k2-WTConv.yaml')

    model.load('yolo11n.pt')

    model.train(
        data=r"C:\Develop\BS\Insulator_System_m\datasets\IDD_yolov11\data.yaml",
        task='detect',
        imgsz=640,
        epochs=300,  # 改进模型建议稍微增加轮次（如150），让新模块充分收敛

        # --- 性能优化（针对你的3060显卡和16G内存） ---
        batch=16,  # 从4调到16。4太小会导致梯度不稳定。
        cache=False,  # 必须为False！你的内存只有16G，开启后会导致磁盘100%卡死。
        workers=4,  # Windows建议设为4，平衡CPU和磁盘读取。

        # --- 科研涨点技巧 ---
        close_mosaic=10,  # 最后10轮关闭Mosaic增强，这是YOLOv8/11提升精度的标准操作。
        optimizer='SGD',  # 如果对调参不熟悉，建议改为 'auto'。
        amp=True,  # 3060支持硬件加速，开启可大幅提速。

        project='runs/train',
        name='yolo11_WTConv_exp',  # 明确命名，方便后续和 Baseline 对比
    )