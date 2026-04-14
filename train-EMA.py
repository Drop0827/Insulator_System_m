import warnings
import os
import pandas as pd
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 1. 加载 EMA 配置文件
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-EMA-P2.yaml')
    model.load('yolo11n.pt') 

    # 2. 开始训练
    model.train(
        data=r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\data.yaml",
        task='detect',
        imgsz=640,
        epochs=200,
        batch=32,
        cache=False,
        workers=4,
        close_mosaic=10,
        optimizer='SGD',
        lr0=0.01,
        amp=True,
        cos_lr=True,
        project='runs/train',
        name='yolo11_EMA_P2_exp',
    )

    # 3. 训练结束后，提取最终数据
    print("\n" + "=" * 50)
    print("训练已结束，正在提取最终验证指标...")

    metrics = model.val()

    # 准备存储最终数据的列表
    final_stats = []

    # 提取汇总行
    final_stats.append({
        'Class': 'all',
        'Precision': metrics.results_dict.get('metrics/precision(B)', 0),
        'Recall': metrics.results_dict.get('metrics/recall(B)', 0),
        'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0)
    })

    # 提取各类别
    names = metrics.names
    for i, name in names.items():
        p, r, ap50, ap95 = metrics.class_result(i)
        final_stats.append({
            'Class': name,
            'Precision': p,
            'Recall': r,
            'mAP50': ap50,
            'mAP50-95': ap95
        })

    # 4. 保存结果
    df = pd.DataFrame(final_stats)
    output_csv = "Final_Metrics_EMA.csv"
    df.to_csv(output_csv, index=False)

    print(f"✅ 数据已成功提取并保存至: {output_csv}")
    print("=" * 50)
