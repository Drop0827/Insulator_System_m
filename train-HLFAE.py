import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载改进后的带有 HLFAE 的模型结构
    # （这里直接指向我们刚刚创建的新配置文件）
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-HLFAE.yaml')

    # 加载预训练权重以加速收敛（YOLO会自动剥离结构不兼容的层）
    model.load('yolo11n.pt')

    # 2. 开始训练
    model.train(
        # 数据集路径对齐您现有的配置
        data=r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\data.yaml",
        task='detect',
        imgsz=640,
        epochs=200,  # 保持 200 轮方便和 WTConv 的实验做公平对比
        
        # --- 性能优化（对齐您 3060 显卡 / 16G 内存的参数） ---
        batch=32,      # 由于网络参数有变化，如果显存（VRAM）爆了可以调回 16
        cache=False,   # 内存16G，务必保持 False 避免虚拟内存拖死系统
        workers=4,     # Windows 下最稳定的工作线程数
        
        # --- 训练涨点技巧（严格对齐 WTConv 以控制变量） ---
        close_mosaic=10,   # 最后10轮关闭Mosaic增强
        optimizer='auto',  # 含有新模块网络的首选
        amp=True,          # 混合精度提速
        cos_lr=True,       # 余弦退火学习率：后期 lr 平滑衰减，帮助新模块权重走出局部最优

        # --- 实验结果保存路径设定 ---
        project='runs/train',
        name='yolo11_HLFAE_exp',  # 独立命名，区分 WTConv 实验
    )
