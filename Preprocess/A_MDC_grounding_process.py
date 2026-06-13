import os
import json
import random
import glob
import re
import numpy as np
import SimpleITK as sitk
from skimage import measure
from PIL import Image


# ==========================================
# 1. 词汇与 Prompt 生成逻辑
# ==========================================
def get_size_adjective(area, is_tumor=False):
    if is_tumor:
        if area < 231: return "tiny"
        elif area < 378: return "small"
        elif area < 553: return "medium"
        elif area < 887: return "large"
        else: return "huge"
    else:
        if area < 611: return "tiny"
        elif area < 1070: return "small"
        elif area < 1526: return "medium"
        elif area < 2230: return "large"
        else: return "huge"

def get_morphology_adjectives(solidity, eccentricity, is_tumor=False):
    """
    基于真实统计学三分位数 (33%, 66%) 的形态学特征判定
    独立处理胰腺和肿瘤的不同物理分布特征
    """
    # 1. 动态获取物理阈值
    if is_tumor:
        sol_33, sol_66 = 0.9520, 0.9647
        ecc_33, ecc_66 = 0.5615, 0.7072
    else:
        sol_33, sol_66 = 0.8730, 0.9399
        ecc_33, ecc_66 = 0.7689, 0.8988

    # 2. 边缘平滑度判定 (Solidity: 越低越不规则，越高越平滑饱满)
    if solidity < sol_33:
        # 低致密度 (粗糙/复杂边缘)
        edge_desc = random.choice(["irregular", "lobulated", "complex"])
    elif solidity < sol_66:
        # 中等致密度 (适中起伏) 
        edge_desc = random.choice(["curved", "wavy", "semi-solid"])
    else:
        # 高致密度 (平滑/边界清晰)
        edge_desc = random.choice(["smooth", "solid", "distinct"])
        
    # 3. 长宽比例判定 (Eccentricity: 越接近1越细长，越接近0越圆润)
    if eccentricity > ecc_66:
        # 极细长 (高离心率)
        shape_desc = random.choice(["elongated", "narrow", "thin"])
    elif eccentricity > ecc_33:
        # 比例适中
        shape_desc = random.choice(["normal-shaped", "typical", "standard"])
    else:
        # 圆润饱满 (低离心率)
        shape_desc = random.choice(["round", "oval", "bulbous"])
        
    return edge_desc, shape_desc

def generate_analysis_list(regions, is_tumor=False, has_tumor_in_slice=False):
    """生成结构化的特征分析字典列表"""
    if not regions:
        return []
        
    analysis_list = []
    for prop, _ in regions:
        size_adj = get_size_adjective(prop.area, is_tumor=is_tumor)
        edge_adj, shape_adj = get_morphology_adjectives(prop.solidity, prop.eccentricity, is_tumor=is_tumor)
        
        analysis = {
            "size": size_adj,
            "edge_solidity": edge_adj,
            "shape_eccentricity": shape_adj
        }
        
        # 如果是胰腺，附加健康状态
        if not is_tumor:
            analysis["health_status"] = "abnormal" if has_tumor_in_slice else "healthy"
            
        analysis_list.append(analysis)
        
    return analysis_list


    


# ==========================================
# 2. 图像坐标提取核心逻辑
# ==========================================
def extract_all_morphology_and_bboxes(binary_mask):
    """【改动】：提取所有连通域对象，并按空间坐标顺序进行排序，不再仅仅返回最大的一块"""
    if np.max(binary_mask) == 0: return []
    labeled_mask = measure.label(binary_mask)
    props = measure.regionprops(labeled_mask)
    
    # 提取所有连通域的属性和 bounding box
    regions = [(p, p.bbox) for p in props]
    
    # 强制按 y_min (bbox[0]) 升序排，若 y_min 相同则按 x_min (bbox[1]) 排
    regions.sort(key=lambda x: (x[1][0], x[1][1]))
    return regions

def get_global_bbox(regions_list):
    """【新功能】：提取多目标边界框并集的最大长方形"""
    if not regions_list: 
        return []
        
    y_mins = [r[1][0] for r in regions_list]
    x_mins = [r[1][1] for r in regions_list]
    y_maxs = [r[1][2] for r in regions_list]
    x_maxs = [r[1][3] for r in regions_list]
    
    # 返回包含所有区域的全局最小 y, 最小 x, 最大 y, 最大 x
    return [min(y_mins), min(x_mins), max(y_maxs), max(x_maxs)]

def pixel_to_grid(pixel_bbox, img_shape, grid_size=32):
    """将真实像素坐标等比例映射到离散网格中"""
    if not pixel_bbox or len(pixel_bbox) == 0: return []
    img_h, img_w = img_shape
    bin_h, bin_w = img_h / grid_size, img_w / grid_size
    y_min, x_min, y_max, x_max = pixel_bbox

    grid_y_min = int(np.clip(y_min // bin_h, 0, grid_size - 1))
    grid_x_min = int(np.clip(x_min // bin_w, 0, grid_size - 1))
    grid_y_max = int(np.clip(y_max // bin_h, 0, grid_size - 1))
    grid_x_max = int(np.clip(x_max // bin_w, 0, grid_size - 1))
    return [grid_y_min, grid_x_min, grid_y_max, grid_x_max]


def process_patient_for_llama(pid, img_nii_path, label_nii_path, out_img_dir):
    """处理单个病人文件"""
    img_itk = sitk.ReadImage(img_nii_path)
    label_itk = sitk.ReadImage(label_nii_path)

    img_array = sitk.GetArrayFromImage(img_itk)
    label_array = sitk.GetArrayFromImage(label_itk)

    vol_min, vol_max = img_array.min(), img_array.max()
    patient_dataset = []
    
    for raw_z in range(img_array.shape[0]):
        slice_label = label_array[raw_z, :, :]
        slice_img = img_array[raw_z, :, :]
        
        if np.max(slice_label) == 0:
            continue
            
        slice_label = np.rot90(slice_label, 2)
        slice_img = np.rot90(slice_img, 2)
        current_shape = slice_label.shape

        if vol_max - vol_min > 1e-6:
            slice_img_uint8 = ((slice_img - vol_min) / (vol_max - vol_min) * 255.0).astype(np.uint8)
        else:
            slice_img_uint8 = np.zeros_like(slice_img, dtype=np.uint8)

        png_filename = f"{pid}_z{raw_z:04d}.jpg"
        png_out_path = os.path.join(out_img_dir, png_filename)
        Image.fromarray(slice_img_uint8, mode='L').save(png_out_path)
        abs_img_path = os.path.abspath(png_out_path)

        # 提取二值化标注
        pancreas_region_mask = ((slice_label == 1) | (slice_label == 2)).astype(int)
        tumor_region_mask = (slice_label == 2).astype(int)

        has_pancreas = bool(np.max(pancreas_region_mask) > 0)
        has_tumor = bool(np.max(tumor_region_mask) > 0)

        # 独立提取特征与排序
        p_regions = extract_all_morphology_and_bboxes(pancreas_region_mask) if has_pancreas else []
        t_regions = extract_all_morphology_and_bboxes(tumor_region_mask) if has_tumor else []

        # 1. 独立生成各自的结构化分析字典 (Analysis)
        p_analysis = generate_analysis_list(p_regions, is_tumor=False, has_tumor_in_slice=has_tumor) if has_pancreas else []
        t_analysis = generate_analysis_list(t_regions, is_tumor=True) if has_tumor else []

        # 2. 独立生成各自的并集框 (Global Bbox)
        p_global_pixel_bbox = get_global_bbox(p_regions) if has_pancreas else []
        p_global_grid_bbox = pixel_to_grid(p_global_pixel_bbox, current_shape)

        t_global_pixel_bbox = get_global_bbox(t_regions) if has_tumor else []
        t_global_grid_bbox = pixel_to_grid(t_global_pixel_bbox, current_shape)
        
       
        

# 【核心改动 1】：变相放大文本 Loss，先分析文本，再输出框！
        assistant_dict = {
            "has_target": has_pancreas or has_tumor,
            "pancreas_analysis": p_analysis,
            "pancreas_bbox_grid": p_global_grid_bbox,
            "tumor_analysis": t_analysis,
            "tumor_bbox_grid": t_global_grid_bbox
        }

        user_content = (
            "<image>\nAnalyze this CT slice. First, strictly evaluate the components using these options: "
            "Size from [tiny, small, medium, large, huge]; "
            "Edge solidity from [irregular, lobulated, complex, curved, wavy, semi-solid, smooth, solid, distinct]; "
            "Shape eccentricity from [elongated, narrow, thin, round, oval, bulbous, normal-shaped, typical, standard]. "
            "Then, provide the structured JSON with analysis dictionaries followed by the global bounding box in 32x32 grid format."
        )

        conversation = {
            "conversations": [
                {"from": "human", "value": user_content},
                {"from": "gpt", "value": json.dumps(assistant_dict, ensure_ascii=False)}
            ],
            "images": [abs_img_path]
        }

        patient_dataset.append(conversation)

    return patient_dataset
    

# ==========================================
# 3. 交叉验证与主程序
# ==========================================
def extract_case_id(pid_str):
    match = re.search(r'\d+', pid_str)
    if match: return int(match.group())
    return 0

if __name__ == "__main__":
    # 配置路径：请确保 processed_imagesTr 路径存在
    data_dir = "/root/autodl-tmp"
    raw_img_dir = os.path.join(data_dir, "processed_imagesTr")
    raw_label_dir = os.path.join(data_dir, "processed_labelsTr")
    out_img_dir = os.path.join(data_dir, "llava_slice_imagesTr")

    os.makedirs(out_img_dir, exist_ok=True)

    N_SPLITS = 5
    CURRENT_FOLD = 0

    image_files = sorted(glob.glob(os.path.join(raw_img_dir, "*.nii.gz")))
    patient_ids = [os.path.basename(f).replace(".nii.gz", "") for f in image_files]

    print(f"找到 {len(patient_ids)} 个病人文件。开始切割与转换...")

    unified_patients_data = {}

    for pid in patient_ids:
        print(f"正在处理 {pid} ...")
        img_nii = os.path.join(raw_img_dir, f"{pid}.nii.gz")
        label_nii = os.path.join(raw_label_dir, f"{pid}.nii.gz")

        if not os.path.exists(label_nii):
            continue

        # 【改动说明】：移除了 process_patient_for_llama 调用中的 target_size 参数
        patient_data = process_patient_for_llama(pid, img_nii, label_nii, out_img_dir)
        if patient_data:
            unified_patients_data[pid] = patient_data

    train_data, val_data = [], []

    for pid, conversations in unified_patients_data.items():
        case_id = extract_case_id(pid)
        if (case_id % N_SPLITS) == CURRENT_FOLD:
            val_data.extend(conversations)
        else:
            train_data.extend(conversations)

    train_out_path = os.path.join(data_dir, f"llava_train_fold{CURRENT_FOLD}.json")
    val_out_path = os.path.join(data_dir, f"llava_val_fold{CURRENT_FOLD}.json")

    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(val_out_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"数据处理完毕！")
    print(f"训练集样本: {len(train_data)} 张有效切片 | 验证集样本: {len(val_data)} 张有效切片")
    print(f"JSON已保存至: {train_out_path}")
    print("=" * 50)