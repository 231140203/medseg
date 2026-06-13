import os
import glob
import json
import numpy as np
import SimpleITK as sitk


def crop_xy_around_pancreas_no_padding(img, lbl, target_h=320, target_w=360):
    """
    智能 XY 平面裁剪：优先追踪胰腺，遇边界平移限制，不填0。
    """
    d, h, w = img.shape

    # 极端兜底 (如果预处理后原图实在太小)
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_width = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        img = np.pad(img, pad_width, mode='constant', constant_values=0)
        lbl = np.pad(lbl, pad_width, mode='constant', constant_values=0)
        h, w = img.shape[1], img.shape[2]

    # 寻找胰腺中心
    xy_projection = np.any(lbl > 0, axis=0)
    if np.any(xy_projection):
        y_indices, x_indices = np.where(xy_projection)
        center_y = (np.min(y_indices) + np.max(y_indices)) // 2
        center_x = (np.min(x_indices) + np.max(x_indices)) // 2
    else:
        center_y, center_x = h // 2, w // 2

    # 理想起点计算
    y_start = center_y - target_h // 2
    x_start = center_x - target_w // 2

    # 核心平移逻辑 (Clamping防溢出)
    y_start = max(0, min(y_start, h - target_h))
    x_start = max(0, min(x_start, w - target_w))

    # 执行安全裁剪
    img_cropped = img[:, y_start:y_start + target_h, x_start:x_start + target_w]
    lbl_cropped = lbl[:, y_start:y_start + target_h, x_start:x_start + target_w]

    return img_cropped, lbl_cropped, (y_start, x_start)


if __name__ == "__main__":
    # ================= 1. 配置输入输出路径 =================
    # 输入：阶段一/二跑完后的 nii.gz 文件夹 和 对应的总 JSON
    PREPROCESSED_DIR = "/root/autodl-tmp/Data/Task07_Pancreas"
    PREPROCESSED_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_imagesTr"
    PREPROCESSED_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_labelsTr"
    INPUT_JSON_PATH = os.path.join(PREPROCESSED_DIR, "dataset_properties.json")

    # 输出：最终喂给 DataLoader 的 .npy 文件夹 和 更新后的 JSON
    FINAL_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/image_npy"
    FINAL_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/label_npy"
    OUTPUT_JSON_PATH = os.path.join(PREPROCESSED_DIR, "dataset_properties_updated.json")

    os.makedirs(FINAL_IMG_DIR, exist_ok=True)
    os.makedirs(FINAL_LBL_DIR, exist_ok=True)

    # ================= 2. 加载之前生成的 JSON =================
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"找不到属性文件: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        all_properties = json.load(f)

    # ================= 3. 遍历预处理后的图像进行裁剪 =================
    image_files = sorted(glob.glob(os.path.join(PREPROCESSED_IMG_DIR, "*.nii.gz")))

    print(f"开始对 {len(image_files)} 个已预处理图像进行 Z/XY 裁剪转 Numpy...")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        case_id = filename.replace('.nii.gz', '')
        lbl_path = os.path.join(PREPROCESSED_LBL_DIR, filename)

        if not os.path.exists(lbl_path):
            print(f"[{i + 1}] 警告: 找不到标签 {filename}，跳过。")
            continue

        # if case_id not in all_properties:
        #     print(f"[{i + 1}] 警告: JSON中找不到 {case_id} 的信息，跳过。")
        #     continue

        print(f"[{i + 1}/{len(image_files)}] 正在裁剪: {case_id}")

        # 读取已经归一化和重采样好的数据
        img_npy = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
        lbl_npy = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path)).astype(np.uint8)

        # ---------- (A) Z轴放宽裁剪 ----------
        z_indices = np.where(lbl_npy > 0)[0]
        if len(z_indices) == 0:
            print(f"    -> 警告: 标签为空，跳过 {case_id}")
            continue

        z_min, z_max = np.min(z_indices), np.max(z_indices)

        # 前后各保留 8 张连续切片
        z_start = max(0, int(z_min) - 8)
        z_end = min(img_npy.shape[0], int(z_max) + 8 + 1)

        img_z_cropped = img_npy[z_start:z_end, :, :]
        lbl_z_cropped = lbl_npy[z_start:z_end, :, :]

        # ---------- (B) XY轴防溢出裁剪 ----------
        img_final, lbl_final, xy_offset = crop_xy_around_pancreas_no_padding(
            img_z_cropped, lbl_z_cropped, target_h=320, target_w=360
        )

        # ---------- (C) 保存为快速加载的 Numpy 格式 ----------
        np.save(os.path.join(FINAL_IMG_DIR, f"{case_id}_img.npy"), img_final)
        np.save(os.path.join(FINAL_LBL_DIR, f"{case_id}_lbl.npy"), lbl_final)

        # ---------- (D) 补充更新 JSON 中的坐标信息 ----------
        # 记录这次裁剪的位置，这是以后逆向恢复回大图时极其关键的坐标差值！
        if not all_properties.get(case_id, 0):
            all_properties[case_id] = {}
        all_properties[case_id]["z_start"] = z_start
        all_properties[case_id]["z_end"] = z_end
        all_properties[case_id]["y_start"] = int(xy_offset[0])
        all_properties[case_id]["x_start"] = int(xy_offset[1])
        all_properties[case_id]["shape_after_npy_crop"] = img_final.shape

    # ================= 4. 保存补充后的最终 JSON =================
    with open(OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f:
        json.dump(all_properties, f, indent=4, ensure_ascii=False)

    print("\n✅ 所有数据已裁剪为 Numpy！")
    print(f"✅ 更新了坐标信息的 JSON 已保存至: {OUTPUT_JSON_PATH}")