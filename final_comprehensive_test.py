import torch
import warnings
import pandas as pd
from ultralytics import YOLO

# 1. 屏蔽干扰警告
warnings.filterwarnings('ignore')


def get_class_map50(results):
    """从验证结果中提取特定类别的mAP50"""
    # 假设 0: broken, 1: insulator (请根据你data.yaml的顺序核对)
    # results.box.maps 返回每个类的 mAP50-95，我们取 map50
    maps = results.box.all_ap[:, 0]  # 获取所有类的 AP@0.5
    return maps[0], maps[1]  # 返回 broken_ap, insulator_ap


if __name__ == '__main__':
    # 这一行在 Windows 环境下是必须的
    torch.multiprocessing.freeze_support()

    # --- 配置区域：请核对路径 ---
    baseline_path = r"runs/train/insulator_yolo11_final/weights/best.pt"
    wtconv_path = r"runs/train/yolo11_wtconv_exp/weights/best.pt"

    std_data = r"datasets/IDD_yolo11/data.yaml"  # 标准验证集
    fog_data = r"datasets/IDD_yolo11/data_foggy.yaml"  # 雾天验证集

    # 加载模型
    m_base = YOLO(baseline_path)
    m_wt = YOLO(wtconv_path)

    summary_data = []

    # --- 实验 1: 标准环境考核 ---
    print("\n" + "=" * 30 + " 正在进行：标准环境测试 " + "=" * 30)
    res_b_std = m_base.val(data=std_data, imgsz=640, plots=False, verbose=False)
    res_w_std = m_wt.val(data=std_data, imgsz=640, plots=False, verbose=False)

    b_ap_std, i_ap_std = get_class_map50(res_b_std)
    bw_ap_std, iw_ap_std = get_class_map50(res_w_std)

    summary_data.append(["Baseline", "标准清晰", res_b_std.box.map50, b_ap_std, i_ap_std])
    summary_data.append(["WTConv改进", "标准清晰", res_w_std.box.map50, bw_ap_std, iw_ap_std])

    # --- 实验 2: 模拟雾天考核 ---
    print("\n" + "=" * 30 + " 正在进行：模拟雾天测试 " + "=" * 30)
    res_b_fog = m_base.val(data=fog_data, imgsz=640, plots=False, verbose=False)
    res_w_fog = m_wt.val(data=fog_data, imgsz=640, plots=False, verbose=False)

    b_ap_fog, i_ap_fog = get_class_map50(res_b_fog)
    bw_ap_fog, iw_ap_fog = get_class_map50(res_w_fog)

    summary_data.append(["Baseline", "模拟大雾", res_b_fog.box.map50, b_ap_fog, i_ap_fog])
    summary_data.append(["WTConv改进", "模拟大雾", res_w_fog.box.map50, bw_ap_fog, iw_ap_fog])

    # --- 结果展示与统计 ---
    columns = ["模型", "环境", "全类mAP50", "Broken(缺陷)mAP", "Insulator(正常)mAP"]
    df = pd.DataFrame(summary_data, columns=columns)

    print("\n" + "🏆 绝缘子缺陷检测：综合对比结果表 🏆".center(80, '='))
    print(df.to_string(index=False))
    print("=" * 85)

    # 计算鲁棒性保持率 (雾天mAP / 晴天mAP)
    base_keep = (float(df.iloc[2, 2]) / float(df.iloc[0, 2])) * 100
    wt_keep = (float(df.iloc[3, 2]) / float(df.iloc[1, 2])) * 100

    print(f"\n📊 鲁棒性深度分析报告:")
    print(f"1. 基准模型环境适应率: {base_keep:.2f}% (分数从 {df.iloc[0, 2]:.3f} 跌至 {df.iloc[2, 2]:.3f})")
    print(f"2. 改进模型环境适应率: {wt_keep:.2f}% (分数从 {df.iloc[1, 2]:.3f} 跌至 {df.iloc[3, 2]:.3f})")

    if wt_keep > base_keep:
        print(
            f"3. 结论验证：改进后的 WTConv 模型在恶劣天气下表现出更强的生命力，鲁棒性提升了 {wt_keep - base_keep:.2f}%！")