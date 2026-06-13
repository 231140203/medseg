import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_medical_volumes(img_dir, lbl_dir, out_img_dir, out_lbl_dir, target_size=(512, 512)):
    """
    将图像和标签统一缩放到目标尺寸，并严格控制插值算法
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 找出所有包含标签的文件
    npy_files = [f for f in os.listdir(lbl_dir) if f.endswith('.npy')]
    print(f"🚀 开始上采样 {len(npy_files)} 个病人的数据至 {target_size[0]}x{target_size[1]}...")

    for file_name in tqdm(npy_files):
        img_path = os.path.join(img_dir, file_name)
        lbl_path = os.path.join(lbl_dir, file_name)
        
        # 加载 3D 体素数据
        volume_img = np.load(img_path)
        volume_lbl = np.load(lbl_path)

        new_img_slices = []
        new_lbl_slices = []

        # 逐层切片进行二维 Resize
        for z in range(volume_img.shape[0]):
            img_slice = volume_img[z]
            lbl_slice = volume_lbl[z]

            # 1. 缩放图像：使用双三次插值 (INTER_CUBIC) 保留最佳视觉细节
            # 如果你的原图是浮点型，请确保它在这个过程中没有溢出
            resized_img = cv2.resize(img_slice, target_size, interpolation=cv2.INTER_CUBIC)

            # 2. 缩放标签：【极其关键】必须使用最近邻插值 (INTER_NEAREST)
            # 这样保证缩放后的标签依然只有纯粹的 0, 1, 2，绝对不会出现小数
            resized_lbl = cv2.resize(lbl_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

            new_img_slices.append(resized_img)
            new_lbl_slices.append(resized_lbl)

        # 转回 numpy 数组
        new_volume_img = np.array(new_img_slices)
        new_volume_lbl = np.array(new_lbl_slices)

        # 保存到新的文件夹
        np.save(os.path.join(out_img_dir, file_name), new_volume_img)
        np.save(os.path.join(out_lbl_dir, file_name), new_volume_lbl)

    print("🎉 上采样全部完成！")

if __name__ == "__main__":
    # 请替换为你的原始 256x256 数据路径
    ORIGINAL_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_imagesTr"
    ORIGINAL_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_labelsTr"

    # 新的高分辨率 512x512 数据存放路径
    NEW_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_imagesTr_512"
    NEW_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_labelsTr_512"

    resize_medical_volumes(ORIGINAL_IMG_DIR, ORIGINAL_LBL_DIR, NEW_IMG_DIR, NEW_LBL_DIR, target_size=(512, 512))