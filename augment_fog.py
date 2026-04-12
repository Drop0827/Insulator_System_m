import os
import cv2
import numpy as np
import random
import shutil

train_img_dir = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\train\images"
train_lbl_dir = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator\train\labels"

def add_fog_basic(image):
    # 随机透明度，控制雾的浓度
    alpha = random.uniform(0.35, 0.65)
    # 创建灰白色的雾气图层
    fog = np.full(image.shape, 210, dtype=np.uint8) 
    return cv2.addWeighted(image, 1 - alpha, fog, alpha, 0)

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
    foggy_img = add_fog_basic(img)
    
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
