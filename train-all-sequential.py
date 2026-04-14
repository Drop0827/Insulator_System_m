import warnings
import os
import pandas as pd
import torch
import gc
from ultralytics import YOLO

warnings.filterwarnings('ignore')

def run_training_experiment(yaml_file, model_name, data_yaml):
    """
    封装单个训练实验的函数，包含内存清理
    """
    print(f"\n\n{'#'*60}")
    print(f"开始实验: {model_name}")
    print(f"配置文件: {yaml_file}")
    print(f"{'#'*60}\n")

    model = None
    try:
        # 1. 初始化模型
        model = YOLO(yaml_file)
        model.load('yolo11n.pt') 

        # 2. 训练 (降低了 batch 和 workers)
        model.train(
            data=data_yaml,
            task='detect',
            imgsz=640,
            epochs=200,
            batch=8,         # <--- 从 32 降到 8，如果还溢出可以试 4
            cache=False,
            workers=2,       # <--- 降低线程数减少内存峰值
            close_mosaic=10,
            optimizer='SGD',
            lr0=0.01,
            amp=True,
            cos_lr=True,
            project='runs/train',
            name=model_name,
            device=0,        # 指定设备
        )

        # 3. 提取指标
        print(f"正在提取 {model_name} 的最终指标...")
        metrics = model.val()
        
        final_stats = []
        final_stats.append({
            'Model': model_name,
            'Class': 'all',
            'Precision': metrics.results_dict.get('metrics/precision(B)', 0),
            'Recall': metrics.results_dict.get('metrics/recall(B)', 0),
            'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        })
        
        # 保存结果
        df = pd.DataFrame(final_stats)
        df.to_csv(f"Results_{model_name}.csv", index=False)
        print(f"✅ {model_name} 实验完成。")

    finally:
        # 4. 强制清理显存 (关键步骤)
        if model:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"🧹 已清理 {model_name} 占用的显存。")

if __name__ == '__main__':
    DATA_PATH = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\data.yaml"
    
    tasks = [
        ['ultralytics/cfg/models/11/yolo11-C3k2-EMA-P2.yaml', 'yolo11_EMA'],
        ['ultralytics/cfg/models/11/yolo11-C3k2-BiFormer-P2.yaml', 'yolo11_BiFormer'],
    ]

    for yaml, name in tasks:
        if not os.path.exists(yaml):
            print(f"⚠️ 警告: 配置文件 {yaml} 不存在，跳过。")
            continue

        try:
            run_training_experiment(yaml, name, DATA_PATH)
        except Exception as e:
            print(f"❌ 实验 {name} 遇到不可恢复错误: {e}")
            # 出错后也尝试清理一次
            gc.collect()
            torch.cuda.empty_cache()
            continue 

    print("\n所有任务已尝试顺序执行完毕！")
