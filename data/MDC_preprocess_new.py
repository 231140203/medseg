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
from MDC_preprocess import NumpyEncoder


# ==========================================
# 1. 面积分档与大小形容词
# ==========================================
def get_dynamic_point_count(area, is_tumor=False):
    if is_tumor:
        if area < 1010:
            return 24
        elif area < 1800:
            return 32
        elif area < 3030:
            return 48
        elif area < 5890:
            return 64
        else:
            return 80
    else:
        if area < 2905:
            return 32
        elif area < 5015:
            return 48
        elif area < 7940:
            return 64
        elif area < 12030:
            return 80
        else:
            return 96


def get_size_adjective(area, is_tumor=False):
    if is_tumor:
        if area < 1010:
            return "tiny"
        elif area < 1800:
            return "small"
        elif area < 3030:
            return "medium"
        elif area < 5890:
            return "large"
        else:
            return "huge"
    else:
        if area < 2905:
            return "tiny"
        elif area < 5015:
            return "small"
        elif area < 7940:
            return "medium"
        elif area < 12030:
            return "large"
        else:
            return "huge"


def get_article(word):
    """根据首字母返回 a 或 an"""
    return "an" if word[0].lower() in "aeiou" else "a"


# ==========================================
# 2. 多边形与形态学联合提取
# ==========================================
def extract_all_polygons(mask, min_area=15, is_tumor=False):
    """提取多边形，同时通过 CV2 直接计算并返回面积和形态学特征"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons_processed = []
    areas_processed = []
    morphs_processed = []  # 记录形态学词汇表

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # --- 核心：直接用 CV2 计算形态学特征 ---
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 1.0

        eccentricity = 0
        if len(contour) >= 5:  # 拟合椭圆至少需要 5 个点
            try:
                _, (MA, ma), _ = cv2.fitEllipse(contour)
                a, b = max(MA, ma) / 2, min(MA, ma) / 2
                if a > 0: eccentricity = np.sqrt(1 - (b / a) ** 2)
            except:
                pass

        # 匹配原来的形态学逻辑
        if solidity < 0.85:
            morph_options = ["irregular", "curved", "complex"]
        else:
            if eccentricity > 0.80:
                morph_options = ["elongated", "narrow", "thin"]
            elif eccentricity < 0.60:
                morph_options = ["round", "oval"]
            else:
                morph_options = ["solid", "normal-shaped"] if not is_tumor else ["solid", "distinct"]
        # --------------------------------------

        contour_sq = contour.squeeze()
        if len(contour_sq.shape) == 1:
            contour_sq = contour_sq.reshape(-1, 2)

        target_num_points = get_dynamic_point_count(area, is_tumor=is_tumor)

        diffs = np.diff(contour_sq, axis=0, append=contour_sq[:1])
        dists = np.linalg.norm(diffs, axis=1)
        cumulative_dists = np.insert(np.cumsum(dists), 0, 0)
        total_length = cumulative_dists[-1]

        if total_length == 0: continue

        query_dists = np.linspace(0, total_length, target_num_points, endpoint=False)
        x_interp = interp1d(cumulative_dists, np.append(contour_sq[:, 0], contour_sq[0, 0]))
        y_interp = interp1d(cumulative_dists, np.append(contour_sq[:, 1], contour_sq[0, 1]))

        poly_2d = np.column_stack((x_interp(query_dists), y_interp(query_dists)))
        poly_2d = np.round(poly_2d).astype(int)
        poly_1d_list = poly_2d.flatten().tolist()

        # (假设你在此处调用了 is_clockwise 和 reorder_points)
        if not is_clockwise(poly_1d_list):
            poly_1d_list = revert_direction(poly_1d_list)
        poly_1d_list = reorder_points(poly_1d_list)

        polygons_processed.append(poly_1d_list)
        areas_processed.append(area)
        morphs_processed.append(morph_options)

    # =======================================================
    # 强制空间排序：只按左上到右下排序！完美保持与文本生成的 1:1 对齐
    # =======================================================
    if len(polygons_processed) > 0:
        zipped = list(zip(polygons_processed, areas_processed, morphs_processed))
        # 仅基于起点坐标的距离和相对位置排序
        zipped = sorted(zipped, key=lambda p: (p[0][0] ** 2 + p[0][1] ** 2, p[0][0], p[0][1]))
        polygons_processed, areas_processed, morphs_processed = zip(*zipped)

    return list(polygons_processed), list(areas_processed), list(morphs_processed)


# ==========================================
# 3. 严格按空间顺序生成的 Prompt 引擎
# ==========================================
def generate_sequential_prompts(areas, morphs, is_tumor=False, has_tumor_in_slice=False, num_variations=3):
    """
    完全放弃合并同类项，严格按照从左到右的空间顺序，
    为每一个目标生成独立描述，通过逗号串联，实现绝对的一一映射。
    """
    if not areas: return []

    prefixes = ["Segment", "Find", "Outline", "Delineate", "Extract"]
    prompts = []

    for _ in range(num_variations):
        phrases = []

        # 严格按照 zip 排序后的顺序遍历
        for i in range(len(areas)):
            size_adj = get_size_adjective(areas[i], is_tumor)
            morph_adj = random.choice(morphs[i])

            if is_tumor:
                noun = random.choice(["tumor", "mass", "lesion", "neoplasm"])
                desc = f"{size_adj} {morph_adj} {noun}"
            else:
                base_noun = random.choice(["pancreas", "pancreas organ", "pancreatic tissue"])
                status = random.choice(['abnormal', 'unhealthy']) if has_tumor_in_slice else random.choice(
                    ["normal", "healthy"])
                # 英语习惯：size -> shape -> condition -> noun (例如: large round abnormal pancreas)
                desc = f"{size_adj} {morph_adj} {status} {base_noun}"

            article = get_article(desc)
            phrases.append(f"{article} {desc}")

        # 将所有的单一描述串联起来
        if len(phrases) == 1:
            combined_desc = phrases[0]
        else:
            combined_desc = " and ".join(phrases)

        prefix = random.choice(prefixes)
        prompts.append(f"{prefix} {combined_desc}".lower())

    return prompts


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

            # 2. 提取多边形
            pancreas_polys, panc_areas, panc_morphs = extract_all_polygons(panc_mask, is_tumor=False) if has_panc else ([],[],[])
            tumor_polys, tumor_areas, tumor_morphs = extract_all_polygons(tumor_mask, is_tumor=True) if has_tumor else ([],[],[])

            # 3. 【核心策略】：切片裂变存储 (将胰腺和肿瘤拆分为独立样本)
            if has_panc and len(pancreas_polys) > 0:
                panc_prompts = generate_sequential_prompts(panc_areas, panc_morphs, is_tumor=False,
                                                           has_tumor_in_slice=has_tumor)
                patient_slices_info.append({
                    "relative_z": int(z),
                    "prompts": panc_prompts,
                    "task": "pancreas",
                    "polygons": pancreas_polys
                })

            if has_tumor and len(tumor_polys) > 0:
                tumor_prompts = generate_sequential_prompts(tumor_areas, tumor_morphs, is_tumor=True,
                                                           has_tumor_in_slice=has_tumor)
                patient_slices_info.append({
                    "relative_z": int(z),
                    "prompts": tumor_prompts,
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
    json_path = os.path.join(output_dir, "MDC512_new_annotations.json")
    pickle_path = os.path.join(output_dir, "MDC512_new_annotations.p")

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