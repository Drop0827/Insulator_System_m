import cv2
import numpy as np
import os
import shutil

# 路径设置
img_dir = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\train\images"
artifact_dir = r"C:\Users\27207\.gemini\antigravity\brain\00bc15cb-3abe-489a-8762-a41bfb66e530\artifacts"
mosaic_src = r"c:\Develop\BS\Insulator_System_m\runs\train\yolo11_PConv_exp\train_batch0.jpg"

def generate_synthetic_fog(image, t=0.5, A=0.8):
    img_float = image.astype(np.float32) / 255.0
    foggy_img = img_float * t + A * (1 - t)
    foggy_img = np.clip(foggy_img, 0, 1)
    return (foggy_img * 255).astype(np.uint8)

# 1. 选取一张原始图像
samples = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
if samples:
    sample_img_path = os.path.join(img_dir, samples[0])
    # 处理中文路径
    img = cv2.imdecode(np.fromfile(sample_img_path, dtype=np.uint8), -1)
    
    if img is not None:
        # 保存原始图 (a)
        cv2.imwrite(os.path.join(artifact_dir, "report_a_clear.jpg"), img)
        
        # 生成并保存雾化图 (b)
        foggy = generate_synthetic_fog(img)
        cv2.imwrite(os.path.join(artifact_dir, "report_b_foggy.jpg"), foggy)
        print("Created clear and foggy sample images.")

# 2. 复制 Mosaic 图片 (c)
if os.path.exists(mosaic_src):
    shutil.copy2(mosaic_src, os.path.join(artifact_dir, "report_c_mosaic.jpg"))
    print("Copied mosaic sample image.")
else:
    print("Mosaic source not found.")
