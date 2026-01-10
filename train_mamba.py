import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载 MLLA 架构
    model = YOLO('yolo11-MLLA.yaml')

    # 2. 加载权重
    model.load('yolo11n.pt')

    # 3. 开启训练
    model.train(
        data=r"C:\Develop\BS\Insulator_System_m\datasets\IDD_yolov11\data.yaml",
        task='detect',
        imgsz=640,
        epochs=200,
        batch=16,
        workers=4,

        # --- 核心修复：解决 ComplexHalf 报错 ---
        deterministic=False,  # 必须设为 False，允许非确定性计算以适配 Mamba 复数运算

        # --- 其他优化配置 ---
        rect=False,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        close_mosaic=20,
        amp=True,

        project='runs/train',
        name='yolo11_MLLA_exp_fixed',
    )