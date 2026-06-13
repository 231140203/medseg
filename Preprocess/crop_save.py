import os
import glob
import json
import numpy as np
import SimpleITK as sitk

def crop_z_and_replace(img_dir, lbl_dir, margin=8):
    """
    读取预处理后的图像和标签，保留包含胰腺的 Z 轴范围并上下各外扩 margin 张切片。
    覆盖保存原文件以节省空间，并将裁剪坐标记录到 JSON 中。
    """
    # 初始化用于记录信息的字典
    crop_records = {}
    json_path = os.path.join(img_dir, "z_crop_records.json")

    # 获取所有的 .nii.gz 图像文件
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    
    if not img_files:
        print(f"❌ 在 {img_dir} 中未找到任何 .nii.gz 文件！")
        return

    print(f"🔍 找到 {len(img_files)} 个病例，开始执行 Z 轴裁剪并替换...\n")

    for i, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, filename)

        if not os.path.exists(lbl_path):
            print(f"[{i+1}/{len(img_files)}] ⚠️ 警告：找不到对应的标签文件 {filename}，跳过。")
            continue

        # 1. 读取图像和标签
        img_itk = sitk.ReadImage(img_path)
        lbl_itk = sitk.ReadImage(lbl_path)

        # SimpleITK 得到的 numpy 数组形状为 (Z, Y, X)
        img_npy = sitk.GetArrayFromImage(img_itk)
        lbl_npy = sitk.GetArrayFromImage(lbl_itk)

        original_z, h, w = img_npy.shape

        # 2. 寻找包含胰腺（label > 0）的切片索引
        z_indices = np.where(lbl_npy > 0)[0]
        
        if len(z_indices) == 0:
            print(f"[{i+1}/{len(img_files)}] ⚠️ 警告：{filename} 的标签中没有任何胰腺前景，保留原图。")
            continue

        z_min = np.min(z_indices)
        z_max = np.max(z_indices)

        # 3. 计算放宽 margin（默认 8）后的起止切片，并严格防止越界
        z_start = max(0, int(z_min) - margin)
        z_end = min(original_z, int(z_max) + margin + 1)

        # 4. 执行 NumPy 数组裁剪
        img_cropped_npy = img_npy[z_start:z_end, :, :]
        lbl_cropped_npy = lbl_npy[z_start:z_end, :, :]

        # 5. 将裁剪后的数组转换回 SimpleITK 图像对象
        new_img_itk = sitk.GetImageFromArray(img_cropped_npy)
        new_lbl_itk = sitk.GetImageFromArray(lbl_cropped_npy)

        # ================= 核心：保持医学物理空间的一致性 =================
        # 复制原图的体素间距 (Spacing) 和方向矩阵 (Direction)
        new_img_itk.SetSpacing(img_itk.GetSpacing())
        new_img_itk.SetDirection(img_itk.GetDirection())
        new_lbl_itk.SetSpacing(lbl_itk.GetSpacing())
        new_lbl_itk.SetDirection(lbl_itk.GetDirection())
        
        # 计算 Z 轴裁剪后的“新原点 (Origin)”物理坐标
        # 原图中索引为 (0, 0, z_start) 的点，就是新图的 (0, 0, 0) 点
        new_origin = img_itk.TransformIndexToPhysicalPoint((0, 0, z_start))
        new_img_itk.SetOrigin(new_origin)
        new_lbl_itk.SetOrigin(new_origin)
        # =================================================================

        # 6. 直接覆盖原路径保存（替换）
        sitk.WriteImage(new_img_itk, img_path)
        sitk.WriteImage(new_lbl_itk, lbl_path)

        # 7. 记录到字典中 (给将来的 DataLoader 和逆向还原使用)
        case_id = filename.replace('.nii.gz', '')
        crop_records[case_id] = {
            "filename": filename,
            "original_z_depth": int(original_z),
            "z_start": int(z_start),
            "z_end": int(z_end),
            "cropped_z_depth": int(z_end - z_start)
        }

        # 打印进度日志
        print(f"[{i+1}/{len(img_files)}] ✅ {case_id} | Z轴瘦身: {original_z}层 -> {z_end - z_start}层 (截取区间: {z_start}~{z_end-1})")

    # 8. 循环结束后，统一保存 JSON 记录表
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(crop_records, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*55)
    print(f"🎉 批量裁剪替换完成！释放了大量无用背景空间。")
    print(f"📁 裁剪范围账本已保存至: {json_path}")
    print("="*55)

if __name__ == "__main__":
    # ================= 配置您的文件夹路径 =================
    # 假设您的数据存放在这两个文件夹中
    PREPROCESSED_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_imagesTr"
    PREPROCESSED_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_labelsTr"
    
    # 运行函数，指定 margin 为 8
    crop_z_and_replace(PREPROCESSED_IMG_DIR, PREPROCESSED_LBL_DIR, margin=8)