import os
import shutil

dir_ds1 = r"c:\Develop\BS\Insulator_System_m\datasets\insulator.v1i.yolov11"
dir_ds2 = r"c:\Develop\BS\Insulator_System_m\datasets\IDD_yolo11"
dir_out = r"c:\Develop\BS\Insulator_System_m\datasets\Combined_Insulator"

# Mapping logic
# New classes: 0 -> insulator, 1 -> broken

# DS1 classes: 0 -> damaged insulator, 1 -> good insulator, 2 -> insulator
map_ds1 = {0: 1, 1: 0, 2: 0}

# DS2 classes: 0 -> broken, 1 -> insulator
map_ds2 = {0: 1, 1: 0}

splits = ['train', 'valid', 'test']
for split in splits:
    os.makedirs(os.path.join(dir_out, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_out, split, 'labels'), exist_ok=True)

def process_dataset(ds_dir, ds_map, prefix=""):
    for split in splits:
        img_dir = os.path.join(ds_dir, split, 'images')
        lbl_dir = os.path.join(ds_dir, split, 'labels')
        
        if not os.path.exists(img_dir): continue
        
        for p in os.listdir(img_dir):
            if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # define new name in case of collision
                new_basename = f"{prefix}_{p}"
                
                src_img = os.path.join(img_dir, p)
                dst_img = os.path.join(dir_out, split, 'images', new_basename)
                shutil.copy2(src_img, dst_img)
                
                # Check label
                base_no_ext = os.path.splitext(p)[0]
                src_lbl = os.path.join(lbl_dir, base_no_ext + ".txt")
                dst_lbl = os.path.join(dir_out, split, 'labels', f"{prefix}_{base_no_ext}.txt")
                
                if os.path.exists(src_lbl):
                    with open(src_lbl, "r") as f: lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            new_cls = ds_map.get(cls_id, -1)
                            if new_cls != -1:
                                parts[0] = str(new_cls)
                                new_lines.append(" ".join(parts) + "\n")
                    with open(dst_lbl, "w") as f:
                        f.writelines(new_lines)


print("Processing Dataset 1...")
process_dataset(dir_ds1, map_ds1, prefix="ds1")
print("Processing Dataset 2...")
process_dataset(dir_ds2, map_ds2, prefix="ds2")

# Create data.yaml
dir_out_slash = dir_out.replace('\\', '/')
yaml_content = f"""path: {dir_out_slash}
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['insulator', 'broken']
"""

with open(os.path.join(dir_out, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Data merged successfully into: {dir_out}")
