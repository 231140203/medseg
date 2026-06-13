import cv2
import numpy as np
from scipy.interpolate import interp1d
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points
import random
import pickle
from skimage.measure import label as sk_label, regionprops
from scipy.interpolate import interp1d
import json
import os
from tqdm import tqdm


def get_dynamic_point_count(area, is_tumor=False):
    """
    基于 512x512 高分辨率与填补肿瘤后的最新面积统计进行分档。
    分档原则: 25% / 50% / 75% / 90%
    注意：由于周长变长，基础点数整体上调，以确保边缘细腻度。
    """
    if is_tumor:
        # 🩸 肿瘤 (Tumor) 512尺度统计: 1006 / 1795 / 3025 / 5887
        if area < 1010:
            return 24  # 极小档 (即使是极小肿瘤，在 512 下也有上千像素，24点能画得很圆润)
        elif area < 1800:
            return 32  # 小档
        elif area < 3030:
            return 48  # 中档
        elif area < 5890:
            return 64  # 大档
        else:
            return 80  # 特大档 (Max可达6万+，80个点足以刻画浸润性边缘)

    else:
        # 🫀 胰腺 (Pancreas - 完整版) 512尺度统计: 2902 / 5013 / 7936 / 12025
        if area < 2905:
            return 32  # 极小档 (切片边缘的胰腺截面)
        elif area < 5015:
            return 48  # 小档
        elif area < 7940:
            return 64  # 中档
        elif area < 12030:
            return 80  # 大档
        else:
            return 96  # 特大档 (Max: 65207，96点/192坐标完美贴合巨大器官外轮廓)


def extract_all_polygons(mask, min_area=15, is_tumor=False):
    """提取 Mask 中的所有多边形实例，根据面积自动分档采样并规范化"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons_processed = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(-1, 2)
            
        # =======================================================
        # 1. 动态获取当前实例的采样点数！
        # =======================================================
        target_num_points = get_dynamic_point_count(area, is_tumor=is_tumor)
        
        diffs = np.diff(contour, axis=0, append=contour[:1])
        dists = np.linalg.norm(diffs, axis=1)
        cumulative_dists = np.insert(np.cumsum(dists), 0, 0)
        total_length = cumulative_dists[-1]
        
        if total_length == 0: continue
            
        query_dists = np.linspace(0, total_length, target_num_points, endpoint=False)
        x_interp = interp1d(cumulative_dists, np.append(contour[:, 0], contour[0, 0]))
        y_interp = interp1d(cumulative_dists, np.append(contour[:, 1], contour[0, 1]))
        
        poly_2d = np.column_stack((x_interp(query_dists), y_interp(query_dists)))
        poly_2d = np.round(poly_2d).astype(int)
        
        poly_1d_list = poly_2d.flatten().tolist()

        # 调用原版规范化函数 (笔顺对齐)
        if not is_clockwise(poly_1d_list):
            poly_1d_list = revert_direction(poly_1d_list)
        poly_1d_list = reorder_points(poly_1d_list)

        polygons_processed.append(poly_1d_list)

    # 排序：从左上到右下
    if len(polygons_processed) > 0:
        polygons_processed = sorted(polygons_processed, key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0], p[1]))

    return polygons_processed


# ==========================================
# 3. 形态学 Prompt 生成
# ==========================================
def get_morphology_desc(mask, is_pancreas=True):
    """提取单个 Mask 的形状，返回形容词和名词集合"""
    if is_pancreas:
        nouns = ["pancreas", "pancreas organ", "pancreatic tissue"]
    else:
        nouns = ["tumor", "mass", "lesion", "pancreatic tumor", "neoplasm"]

    regions = regionprops(sk_label(mask))
    if not regions:
        return ["a"], nouns  # 如果没有区域，给个保底

    target_region = max(regions, key=lambda r: r.area)
    ecc = target_region.eccentricity
    sol = target_region.solidity

    if sol < 0.85:
        adjectives = ["an irregular", "a curved", "a complex", "a lobulated"]
    else:
        if ecc > 0.80:
            adjectives = ["an elongated", "a narrow", "a thin"]
        elif ecc < 0.60:
            adjectives = ["a round", "an oval", "a bulbous"]
        else:
            adjectives = ["a solid", "a normal-shaped"] if is_pancreas else ["a solid", "a distinct"]

    return adjectives, nouns


import random


def generate_decoupled_sentences(lbl_slice):
    """
    为单张切片生成解耦的 Prompt。
    针对带肿瘤的胰腺：100% 明确指出其包含病灶及其形态。
    """
    panc_mask = (lbl_slice == 1) | (lbl_slice == 2)
    tumor_mask = (lbl_slice == 2)

    has_panc = np.any(panc_mask)
    has_tumor = np.any(tumor_mask)

    prefixes = ["Segment", "Find", "Outline", "Delineate", "Extract"]
    prompts_dict = {}

    # ----------------------------------------------------
    # 1. 专门针对“胰腺”的 Prompt
    # ----------------------------------------------------
    if has_panc:
        panc_adjs, panc_nouns = get_morphology_desc(panc_mask, is_pancreas=True)
        panc_sentences = []
        for _ in range(3):
            prefix = random.choice(prefixes)

            # 【关键修改】：只要有肿瘤，统一强制生成带病灶描述的胰腺！不再随机！
            if has_tumor:
                panc_desc = random.choice(['abnormal', 'unhealthy', 'bulging'])
                # conj = random.choice(['containing', 'with', 'harboring'])
                # 可以用 等词连接
                sent = f"{prefix} {random.choice(panc_adjs)} {panc_desc} {random.choice(panc_nouns)}"
            else:
                panc_desc = random.choice(["normal", "healthy"])
                sent = f"{prefix} {random.choice(panc_adjs)} {panc_desc} {random.choice(panc_nouns)}"

            panc_sentences.append(sent.lower())
        prompts_dict["pancreas"] = panc_sentences

    # ----------------------------------------------------
    # 2. 专门针对“肿瘤”的 Prompt
    # ----------------------------------------------------
    if has_tumor:
        tumor_adjs, tumor_nouns = get_morphology_desc(tumor_mask, is_pancreas=False)
        tumor_sentences = []
        for _ in range(3):
            prefix = random.choice(prefixes)
            tumor_desc = f"{random.choice(tumor_adjs)} {random.choice(tumor_nouns)}"

            sent = f"{prefix} {tumor_desc}"

            tumor_sentences.append(sent.lower())
        prompts_dict["tumor"] = tumor_sentences

    return prompts_dict


# ==========================================
# 4. JSON 序列化辅助类
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """解决 NumPy 数据类型无法直接 JSON 序列化的问题"""

    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# 5. 主遍历流水线
# ==========================================
def process_dataset(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    global_index_dict = {}

    npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    print(f"🚀 开始处理 {len(npy_files)} 个病人的标签数据...")

    for file_name in tqdm(npy_files):
        patient_id = file_name.replace('.npy', '')
        mask_path = os.path.join(label_dir, file_name)

        volume_label = np.load(mask_path)

        patient_slices_info = []

        # 遍历该病人的每一层切片

        for z in range(volume_label.shape[0]):
            lbl_slice = volume_label[z]

            panc_mask = (lbl_slice == 1) | (lbl_slice == 2)
            tumor_mask = (lbl_slice == 2)

            has_panc = np.any(panc_mask)
            has_tumor = np.any(tumor_mask)

            if not has_panc and not has_tumor:
                continue  # 跳过纯背景切片

            # 1. 生成解耦的 Prompt
            prompts_dict = generate_decoupled_sentences(lbl_slice)

            # 2. 提取多边形
            pancreas_polys = extract_all_polygons(panc_mask, is_tumor=False) if has_panc else []
            tumor_polys = extract_all_polygons(tumor_mask, is_tumor=True) if has_tumor else []

            # 3. 【核心策略】：切片裂变存储 (将胰腺和肿瘤拆分为独立样本)
            if has_panc and len(pancreas_polys) > 0:
                patient_slices_info.append({
                    "relative_z": int(z),
                    "prompts": prompts_dict.get("pancreas", []),
                    "task": "pancreas",
                    "polygons": pancreas_polys
                })

            if has_tumor and len(tumor_polys) > 0:
                patient_slices_info.append({
                    "relative_z": int(z),
                    "prompts": prompts_dict.get("tumor", []),
                    "task": "tumor",
                    "polygons": tumor_polys
                })

        # 将该病人信息存入全局字典
        global_index_dict[patient_id] = {
            "img_path": f"crop_imagesTr_512/{file_name}",
            "mask_path": f"crop_labelsTr_512/{file_name}",
            "slices_info": patient_slices_info
        }

    # ==========================================
    # 6. 数据落盘 (JSON + Pickle)
    # ==========================================
    json_path = os.path.join(output_dir, "MDC512_sep_annotations.json")
    pickle_path = os.path.join(output_dir, "MDC512_sep_annotations.p")

    print(f"\n💾 正在保存 JSON 格式至 {json_path} ...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(global_index_dict, f, indent=2, cls=NumpyEncoder)

    print(f"💾 正在保存 Pickle 格式至 {pickle_path} ...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(global_index_dict, f)

    print("🎉 全部处理完成！数据集构建成功！")


if __name__ == "__main__":
    # 配置你的路径
    LABEL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_labelsTr_512" # 标签数据源路径
    OUTPUT_DIR = "/root/autodl-tmp/Data/Task07_Pancreas" # .p 和 .json 保存的目标路径

    process_dataset(LABEL_DIR, OUTPUT_DIR)