import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格，使其符合学术论文要求
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.style.use('ggplot')

# --- 填入你刚才截图中的真实数据 ---
labels = ['标准清晰环境', '模拟大雾环境']
baseline_map = [0.816, 0.787]
wtconv_map = [0.828, 0.823]

baseline_broken = [0.637, 0.578]
wtconv_broken = [0.661, 0.652]

x = np.arange(len(labels))
width = 0.35

# --- 图 1：总体 mAP50 对比 ---
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_map, width, label='Baseline (YOLO11n)', color='#808080')
rects2 = ax.bar(x + width/2, wtconv_map, width, label='本文改进 (WTConv)', color='#574FF5')

ax.set_ylabel('mAP@0.5')
ax.set_title('不同环境下的全类平均精度(mAP50)对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.7, 0.85) # 聚焦差异
ax.legend()

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('overall_map_comparison.png', dpi=300) # 保存高分辨率图用于论文
plt.show()

# --- 图 2：Broken(缺陷)类别专项对比 ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
rects3 = ax2.bar(x - width/2, baseline_broken, width, label='Baseline (YOLO11n)', color='#A9A9A9')
rects4 = ax2.bar(x + width/2, wtconv_broken, width, label='本文改进 (WTConv)', color='#FF7E79')

ax2.set_ylabel('mAP@0.5')
ax2.set_title('针对“破损(Broken)”缺陷的检测精度对比')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0.5, 0.7)
ax2.legend()

autolabel = lambda rects, axis: [axis.annotate(f'{r.get_height():.3f}',
            xy=(r.get_x() + r.get_width() / 2, r.get_height()),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom') for r in rects]

autolabel(rects3, ax2)
autolabel(rects4, ax2)
fig2.tight_layout()
plt.savefig('broken_class_comparison.png', dpi=300)
plt.show()