import matplotlib.pyplot as plt
import numpy as np
import os

def generate_academic_report(results, save_dir="comparison_reports"):
    """
    results: 列表，每个元素是字典格式，
    如 {"Model": "YOLOv11n", "mAP50(Best)": 0.816, "Latency(ms)": 15.4}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 指标数据准备
    models = [r["Model"] for r in results]
    map_values = [r["mAP50(Best)"] for r in results]
    latency_values = [r["Latency(ms)"] for r in results]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.4
    
    # 🎨 1. mAP50 对比柱状图 (选用科研常用的灰蓝配色)
    colors = ['#8C8C8C', '#5856D6', '#4CAF50', '#FF9800'] # 基准灰色，改进紫色
    bars1 = ax1.bar(x, map_values, width, color=colors[:len(models)])
    ax1.set_title('全类平均精度 (mAP@0.5) 性能对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mAP', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylim(min(map_values)*0.9, min(1.0, max(map_values)*1.1))
    
    # 数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 🚀 2. 推理耗时对照 (Latency)
    bars2 = ax2.bar(x, latency_values, width, color=colors[:len(models)])
    ax2.set_title('推理延迟 (ms/img) 横向对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Latency', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    report_file = os.path.join(save_dir, "final_academic_comparison.png")
    plt.savefig(report_file, dpi=300)
    plt.close()
    return report_file
