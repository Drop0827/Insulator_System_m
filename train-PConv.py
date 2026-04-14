import warnings
import os
import pandas as pd
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 1. 加载模型结构与权重
    # 修改点：确保指向你的 PConv 配置文件
    model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-PConv.yaml')
    model.load('yolo11n.pt')

    # 2. 开始训练
    # 保持你原来的所有参数不变
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
        name='yolo11_WTConv_exp',
    )

    # 3. 训练结束后，提取“红框”中的最终数据
    print("\n" + "=" * 50)
    print("训练已结束，正在提取最终验证指标...")

    # 这一步会加载训练中表现最好的 best.pt 并在验证集上运行
    metrics = model.val()

    # 准备存储最终数据的列表
    final_stats = []

    # 提取 'all' (汇总行)
    final_stats.append({
        'Class': 'all',
        'Precision': metrics.results_dict['metrics/precision(B)'],
        'Recall': metrics.results_dict['metrics/recall(B)'],
        'mAP50': metrics.results_dict['metrics/mAP50(B)'],
        'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)']
    })

    # 提取每个分类 (如 insulator, broken)
    names = metrics.names  # 获取类别名称字典
    for i, name in names.items():
        # metrics.class_result(i) 返回: [precision, recall, map50, map95]
        p, r, ap50, ap95 = metrics.class_result(i)
        final_stats.append({
            'Class': name,
            'Precision': p,
            'Recall': r,
            'mAP50': ap50,
            'mAP50-95': ap95
        })

    # 4. 将数据保存到项目根目录，方便开机后直接查看
    df = pd.DataFrame(final_stats)

    # 保存为 CSV (可用 Excel 打开)
    output_csv = "Final_Metrics_WTConv.csv"
    df.to_csv(output_csv, index=False)

    # 保存为 TXT (方便直接阅读)
    output_txt = "Final_Metrics_WTConv.txt"
    with open(output_txt, "w") as f:
        f.write("YOLOv11 PConv 实验最终数据汇总\n")
        f.write("-" * 60 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "-" * 60)

    print(f"数据已成功提取并保存至: {output_csv} 和 {output_txt}")

    # 5. 执行自动关机
    print("\n任务全部完成！系统将在 60 秒后自动关机...")
    print("如果需要取消关机，请立即按下 Win+R 输入: shutdown /a")
    print("=" * 50)

    # Windows 关机指令
    os.system("shutdown /s /t 60")