import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载带有 P-C3k2（部分卷积）改进的模型结构
    # 参考论文：《基于改进 YOLOv11 的钢轨缺陷检测实验研究》2025
    # 改进点：Neck 中 P4/P5 融合层使用 C3k2_PConv，降低 FLOPs、提升推理速度
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-PConv.yaml')

    # 加载预训练权重（Backbone 完全与基线一致，权重复用率最高）
    model.load('yolo11n.pt')

    # 2. 开始训练
    model.train(
        data=r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\data.yaml",
        task='detect',
        imgsz=640,
        epochs=200,   # 与其他消融实验保持一致（控制变量）

        # --- 硬件参数（对齐 RTX 3060 Laptop / 16G 内存） ---
        batch=32,     # PConv 参数更少，显存压力低于 HLFAE，32 应能稳定运行
        cache=False,  # 内存 16G 不建议开启，防止虚拟内存拖慢训练
        workers=4,    # Windows 多进程稳定值

        # --- 收敛优化 ---
        close_mosaic=10,   # 最后 10 轮关闭 Mosaic，帮助模型稳定收敛
        optimizer='SGD',   # NOTE: PConv 结构对 SGD 收敛更稳定（相比 Adam 不易振荡）
        lr0=0.01,          # 初始学习率，与基线保持一致做公平对比
        amp=True,          # 混合精度加速，对 RTX 系列显卡效果显著
        cos_lr=True,       # 余弦退火学习率：后期 lr 平滑衰减，帮助新模块权重走出局部最优

        # --- 实验结果保存路径 ---
        project='runs/train',
        name='yolo11_PConv_exp',  # 独立命名，便于与 WTConv / HLFAE 实验对比
    )
