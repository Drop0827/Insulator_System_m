import os
import cv2
import numpy as np
import random
import shutil

train_img_dir = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\train\images"
train_lbl_dir = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\train\labels"

def generate_synthetic_fog(image, t_range=(0.4, 0.8), A_range=(0.7, 0.9)):
    """
    基于大气散射模型的绝缘子图像雾化增强算法
    :param image: 输入的高清图像矩阵 (BGR格式)
    :param t_range: 透射率随机区间，控制雾的浓淡
    :param A_range: 大气光强度随机区间
    """
    # 随机生成当前图像的透射率 t 和大气光 A
    t = np.random.uniform(t_range[0], t_range[1])
    A = np.random.uniform(A_range[0], A_range[1])
    
    # 图像归一化至 [0, 1] 以进行浮点运算
    img_float = image.astype(np.float32) / 255.0
    
    # 矩阵向量化运算应用大气散射公式: I(x) = J(x)t + A(1-t)
    foggy_img = img_float * t + A * (1 - t)
    
    # 限制像素范围并还原至 uint8 格式
    foggy_img = np.clip(foggy_img, 0, 1)
    foggy_img = (foggy_img * 255).astype(np.uint8)
    
    return foggy_img

images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Start applying fog augmentation... Found {len(images)} images.")

count = 0
for img_name in images:
    if img_name.startswith("fog_"): 
        continue # 跳过已经雾化的
    
    img_path = os.path.join(train_img_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(train_lbl_dir, base_name + ".txt")
    
    # 解决中文路径读取问题
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None: continue
    
    # 雾化处理
    foggy_img = generate_synthetic_fog(img)
    
    # 写入新图
    new_img_name = "fog_" + img_name
    new_img_path = os.path.join(train_img_dir, new_img_name)
    cv2.imencode('.jpg', foggy_img)[1].tofile(new_img_path)
    
    # 复制同样的标注文件（因为雾化不改变边框位置）
    if os.path.exists(lbl_path):
        new_lbl_path = os.path.join(train_lbl_dir, "fog_" + base_name + ".txt")
        shutil.copy2(lbl_path, new_lbl_path)
    count += 1

print(f"Finished! Augmented {count} images with fog.")
