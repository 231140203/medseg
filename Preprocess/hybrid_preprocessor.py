import os
import glob
import numpy as np
import json
import SimpleITK as sitk
import shutil
from scipy.ndimage import binary_fill_holes, zoom


# ==============================================================================
# 阶段一：nnU-Net 风格的数据清洗与裁剪逻辑
# 参考自 nnunetv2/preprocessing/cropping/cropping.py
# ==============================================================================
def get_bbox_from_mask(mask, outside_value=0):
    """计算三维掩码的 Bounding Box"""
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_nonzero(data, seg=None):
    """寻找非零区域并裁剪 (去除无效背景)"""
    # 生成非零掩码并填充孔洞
    nonzero_mask = (data != 0)
    nonzero_mask = binary_fill_holes(nonzero_mask)

    # 获取 Bounding Box
    bbox = get_bbox_from_mask(nonzero_mask)

    # 执行裁剪 (Z, X, Y)
    slicer = tuple(slice(bbox[i][0], bbox[i][1]) for i in range(3))

    data_cropped = data[slicer]
    seg_cropped = seg[slicer] if seg is not None else None

    return data_cropped, seg_cropped, bbox


# ==============================================================================
# 阶段二：Dual-Task-Seg 风格的 CT 物理灰度处理与分辨率重采样
# ==============================================================================
def process_ct_intensity(data, low_range=-100, high_range=240):
    """针对胰腺和肿瘤的固定 CT 窗位截断与归一化"""
    # 截断到 [-100, 240]
    data = np.clip(data, low_range, high_range)
    # Min-Max 线性归一化到 [0, 1]
    data = (data - low_range) / (high_range - low_range)
    return data.astype(np.float32)


def resample_3d(data, original_spacing, target_spacing, is_label=False):
    """
    统一物理分辨率 (Spacing)
    图像使用三阶样条插值(order=3)，标签使用最近邻插值(order=0)
    """
    # 计算缩放比例 (Spacing 越大，意味着物理范围越广，体素数量需缩小)
    zoom_factors = [orig / targ for orig, targ in zip(original_spacing, target_spacing)]

    order = 0 if is_label else 3
    # 针对图像使用 spline，针对掩码使用 nearest
    resampled_data = zoom(data, zoom_factors, order=order, mode='nearest')

    if is_label:
        resampled_data = np.round(resampled_data).astype(np.uint8)

    return resampled_data


# ==============================================================================
# 核心执行流水线
# ==============================================================================
def run_hybrid_preprocessing(image_path, label_path, out_img_path, out_lbl_path, target_spacing=(1.0, 0.8, 0.8)):
    print(f"Processing: {os.path.basename(image_path)}")

    # 1. 读取数据
    img_itk = sitk.ReadImage(image_path)
    img_npy = sitk.GetArrayFromImage(img_itk)
    original_spacing = img_itk.GetSpacing()[::-1]

    lbl_npy = sitk.GetArrayFromImage(sitk.ReadImage(label_path)) if label_path and os.path.exists(label_path) else None

    original_shape = img_npy.shape
    original_filename = os.path.basename(image_path)
    case_id = original_filename.replace('.nii.gz', '')

    # 2. 裁剪与灰度处理、重采样
    img_npy, lbl_npy, bbox = crop_to_nonzero(img_npy, lbl_npy)
    shape_after_crop = img_npy.shape

    img_npy = process_ct_intensity(img_npy, low_range=-100, high_range=240)
    img_resampled = resample_3d(img_npy, original_spacing, target_spacing, is_label=False)

    if lbl_npy is not None:
        lbl_resampled = resample_3d(lbl_npy, original_spacing, target_spacing, is_label=True)

    # 3. 保存图像
    new_itk_img = sitk.GetImageFromArray(img_resampled)
    new_itk_img.SetSpacing(target_spacing[::-1])
    new_itk_img.SetDirection(img_itk.GetDirection())
    sitk.WriteImage(new_itk_img, out_img_path)

    if lbl_npy is not None:
        new_itk_lbl = sitk.GetImageFromArray(lbl_resampled)
        new_itk_lbl.SetSpacing(target_spacing[::-1])
        new_itk_lbl.SetDirection(img_itk.GetDirection())
        sitk.WriteImage(new_itk_lbl, out_lbl_path)

    # ================= 核心修改：不再保存文件，而是返回字典 =================
    properties = {
        "original_filename": original_filename,
        "original_image_path": os.path.abspath(image_path),
        "original_shape": original_shape,
        "original_spacing": original_spacing,
        "target_spacing": target_spacing,
        "bbox": bbox,
        "shape_after_crop": shape_after_crop
    }

    return case_id, properties


if __name__ == "__main__":
    RAW_DATA_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/imagesTr"
    RAW_LABEL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/labelsTr"
    OUT_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_imagesTr"
    OUT_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_labelsTr"
    OUT_DIR = '/root/autodl-tmp/Data/Task07_Pancreas'
    DEST_DIR = "/root/data/Task07_Pancreas/imagesTr"

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LBL_DIR, exist_ok=True)

    TARGET_SPACING = (1.0, 0.8, 0.8)

    # 初始化一个全局大字典
    all_dataset_properties = {}

    image_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.nii.gz"))
    for img_path in image_files:
        filename = os.path.basename(img_path)
        lbl_path = os.path.join(RAW_LABEL_DIR, filename)

        out_img_path = os.path.join(OUT_IMG_DIR, filename)
        out_lbl_path = os.path.join(OUT_LBL_DIR, filename)

        # 接收返回值
        case_id, props = run_hybrid_preprocessing(img_path, lbl_path, out_img_path, out_lbl_path, TARGET_SPACING)

        # 将单个病例的属性存入大字典
        all_dataset_properties[case_id] = props

        dst_path = os.path.join(DEST_DIR, filename)

        print(f"🚚 移动: {filename} -> {DEST_DIR}")

        shutil.move(img_path, dst_path)
    # ================= 循环结束后，一次性保存总的 JSON =================
    dataset_json_path = os.path.join(OUT_DIR, "dataset_properties.json")
    with open(dataset_json_path, 'a') as f:
        json.dump(all_dataset_properties, f, indent=4)

    print(f"\n✅ All done! Dataset properties saved to: {dataset_json_path}")