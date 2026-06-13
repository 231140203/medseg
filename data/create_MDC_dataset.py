import os
import cv2
import json
import numpy as np
import random
import nibabel as nib
from tqdm import tqdm
import pickle
from skimage.measure import label as sk_label, regionprops
from scipy.interpolate import interp1d
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points

# ==========================================
# 1. 核心数学规范化函数 (保留 PolyFormer 精华)
# ==========================================
"""
def is_clockwise(poly):
    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += (poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1])
    return area < 0

def revert_direction(poly):
    return poly[::-1]

def reorder_points(poly):
    distances = np.sum(poly**2, axis=1)
    min_idx = np.argmin(distances)
    return np.roll(poly, shift=-min_idx, axis=0)
"""

# ==========================================
# 2. 【关键改进】支持自定义点数的提取函数
# ==========================================
def extract_all_polygons(mask, target_num_points, min_area=15):
    """提取 Mask 中的所有多边形实例，进行自定义点数的重采样并规范化"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons_processed = []
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
            
        contour = contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(-1, 2)
            
        # 1. 按指定的目标点数进行等距重采样 (例如：胰腺64，肿瘤32)
        diffs = np.diff(contour, axis=0, append=contour[:1])
        dists = np.linalg.norm(diffs, axis=1)
        cumulative_dists = np.insert(np.cumsum(dists), 0, 0)
        total_length = cumulative_dists[-1]
        
        if total_length == 0: continue
            
        query_dists = np.linspace(0, total_length, target_num_points, endpoint=False)
        x_interp = interp1d(cumulative_dists, np.append(contour[:, 0], contour[0, 0]))
        y_interp = interp1d(cumulative_dists, np.append(contour[:, 1], contour[0, 1]))
        # poly_2d 此时是 (N, 2) 的 NumPy 数组
        poly_2d = np.column_stack((x_interp(query_dists), y_interp(query_dists)))
        poly_2d = np.round(poly_2d).astype(int)
        # =======================================================
        # 【关键适配】：转换为 1D List 以迎合 PolyFormer 原版函数
        # =======================================================
        poly_1d_list = poly_2d.flatten().tolist()

        # 调用原版规范化函数
        if not is_clockwise(poly_1d_list):
            poly_1d_list = revert_direction(poly_1d_list)
        poly_1d_list = reorder_points(poly_1d_list)

        # 此时的 poly_1d_list 是一个形如 [x1, y1, x2, y2, ...] 的标准一维列表
        polygons_processed.append(poly_1d_list)

        # 多实例空间排序：因为现在是 1D 列表，x1, y1 分别是 [0] 和 [1]
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


def generate_joint_sentences(lbl_slice):
    """
    为包含胰腺和肿瘤的单张切片，生成 3 句多样化的联合 Prompt
    """
    # 假设标签 1 是胰腺，2 是肿瘤
    # 注意：计算胰腺形态时，要包含肿瘤区域
    panc_mask = (lbl_slice == 1) | (lbl_slice == 2)
    tumor_mask = (lbl_slice == 2)

    has_tumor = np.any(tumor_mask)

    panc_adjs, panc_nouns = get_morphology_desc(panc_mask, is_pancreas=True)
    if has_tumor:
        tumor_adjs, tumor_nouns = get_morphology_desc(tumor_mask, is_pancreas=False)

    # 用于连接胰腺和肿瘤的连词
    connectors = ["containing", "with", "which includes", "harboring"]
    # 纯分割任务的前缀
    prefixes = ["Segment", "Find", "Outline", "Delineate", "Extract"]

    sentences = []

    for _ in range(3):
        prefix = random.choice(prefixes)
        panc_desc = f"{random.choice(panc_adjs)} {random.choice(panc_nouns)}"

        if has_tumor:
            conn = random.choice(connectors)
            tumor_desc = f"{random.choice(tumor_adjs)} {random.choice(tumor_nouns)}"

            # 组合句型 1: Segment an elongated pancreas containing a round tumor
            # 组合句型 2: Find a round mass inside a curved pancreatic tissue

            if random.random() > 0.5:
                # 正常语序：胰腺 -> 肿瘤
                sent = f"{prefix} {panc_desc} {conn} {tumor_desc}"
            else:
                # 倒装语序：肿瘤 -> 胰腺 (迫使大模型更深刻地理解介词)
                sent = f"{prefix} {tumor_desc} inside {panc_desc}"
        else:
            # 只有胰腺
            sent = f"{prefix} {panc_desc}"

        # 统一转为小写 (推荐)
        sentences.append(sent.lower())

    return sentences


# --- 测试用例 ---
# 模拟一个标签切片
# fake_lbl = np.zeros((256, 256))
# fake_lbl[100:150, 100:150] = 1 # 假胰腺
# fake_lbl[120:130, 120:130] = 2 # 假肿瘤
# print(generate_joint_sentences(fake_lbl))
# 输出示例:
# ['extract an oval pancreas organ which includes a solid lesion',
#  'segment a solid mass inside an oval pancreatic tissue',
#  'outline an oval pancreas with a distinct tumor']

def get_3d_patient_crop_params(lbl_volume, target_h=256, target_w=256, max_jitter=30):
    """
    计算患者全局的 3D 裁剪参数，支持受控的随机抖动。
    lbl_volume: (H, W, Z) 3D 标签数组
    max_jitter: 允许中心点随机偏移的最大像素数

    返回: y_start, x_start (整个序列共享这一组坐标)
    """
    h, w, _ = lbl_volume.shape

    # 假设标签 1 是胰腺，2 是肿瘤
    valid_mask = (lbl_volume == 1) | (lbl_volume == 2)

    # 如果全图都没有胰腺（理论上在传进此函数前已过滤），返回中心裁剪
    if not np.any(valid_mask):
        return max(0, h // 2 - target_h // 2), max(0, w // 2 - target_w // 2)

    # 1. 寻找 3D 胰腺在 XY 平面的全局投影边界
    # 这样能保证不管胰腺在 Z 轴怎么游走，它的 XY 极值点都在视野内
    xy_projection = np.any(valid_mask, axis=2)
    y_indices, x_indices = np.where(xy_projection)

    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)

    # 2. 计算全局中心
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2

    # 3. 引入随机抖动 (Jitter)
    # 计算安全的抖动范围，确保胰腺不会被切出 256x256 的框
    # 胰腺的 y 高度 / 2 必须小于 target_h / 2
    safe_y_jitter = max(0, (target_h // 2) - ((max_y - min_y) // 2) - 5)  # 留 5 像素安全边距
    safe_x_jitter = max(0, (target_w // 2) - ((max_x - min_x) // 2) - 5)

    # 限制最大抖动幅度
    act_y_jitter = min(safe_y_jitter, max_jitter)
    act_x_jitter = min(safe_x_jitter, max_jitter)

    jitter_y = random.randint(-act_y_jitter, act_y_jitter) if act_y_jitter > 0 else 0
    jitter_x = random.randint(-act_x_jitter, act_x_jitter) if act_x_jitter > 0 else 0

    final_center_y = center_y + jitter_y
    final_center_x = center_x + jitter_x

    # 4. 理想起点计算
    y_start = final_center_y - target_h // 2
    x_start = final_center_x - target_w // 2

    # 5. 边界防溢出裁剪 (Clamping)
    # 这一步极其重要，防止在图像边缘加上偏移后，起始点变成负数或越出右下角
    y_start = max(0, min(y_start, h - target_h))
    x_start = max(0, min(x_start, w - target_w))

    return y_start, x_start

# ==========================================
# 工具类：处理 NumPy 数据类型的 JSON 序列化
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# ==========================================
# 5. 主流程：生成 .npy Volume 与全局 JSON
# ==========================================
def process_data_to_volumes(img_dir, label_dir, output_dir, pts_pancreas=64, pts_tumor=32, target_size=256):
    vol_img_dir = os.path.join(output_dir, "crop_imagesTr")
    vol_mask_dir = os.path.join(output_dir, "crop_labelsTr")
    os.makedirs(vol_img_dir, exist_ok=True)
    os.makedirs(vol_mask_dir, exist_ok=True)

    # 【核心修改】：现在的 json_data 是一个大字典，以 patient_id 为 Key
    global_index_dict = {}

    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])

    for filename in tqdm(files, desc="Processing Volumes"):
        patient_id = filename.replace('.nii.gz', '')
        # print("Loading:", filename)
        img_data = nib.load(os.path.join(img_dir, filename)).get_fdata()
        lbl_data = nib.load(os.path.join(label_dir, filename)).get_fdata()

        if not np.any((lbl_data == 1) | (lbl_data == 2)): continue

        min_val = img_data.min()
        max_val = img_data.max()
        if max_val - min_val > 1e-6:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = np.zeros_like(img_data)
        img_data = img_data.astype(np.float32)

        img_data = np.fliplr(np.rot90(img_data, k=1))
        lbl_data = np.fliplr(np.rot90(lbl_data, k=1))

        y_start, x_start = get_3d_patient_crop_params(lbl_data, target_size, target_size)

        patient_slices_rgb = []
        patient_slices_mask = []
        patient_slices_info = []

        z_dim = img_data.shape[2]
        relative_z = 0  # 记录在保存的 numpy 数组中的有效层索引

        for z in range(z_dim):
            slice_lbl = lbl_data[y_start:y_start + target_size, x_start:x_start + target_size, z]
            slice_img = img_data[y_start:y_start + target_size, x_start:x_start + target_size, z]

            if 1 not in slice_lbl: continue

            panc_mask = (slice_lbl == 1).astype(np.uint8)
            tumor_mask = (slice_lbl == 2).astype(np.uint8)

            panc_polys = extract_all_polygons(panc_mask, pts_pancreas)
            tumor_polys = extract_all_polygons(tumor_mask, pts_tumor)

            if not panc_polys: continue

            # 准备存入 Volume 的数据 (旋转对齐)
            img_rgb = cv2.merge([slice_img, slice_img, slice_img])

            patient_slices_rgb.append(img_rgb)
            patient_slices_mask.append(slice_lbl)

            prompts = generate_joint_sentences(slice_lbl)

            patient_slices_info.append({
                "relative_z": relative_z,  # 在 npy 数组中的索引 (0, 1, 2...)
                "original_z": z,  # 在原始 NIfTI 中的层数
                "prompts": prompts,
                "pancreas_polygons": panc_polys,
                "tumor_polygons": tumor_polys
            })
            relative_z += 1

        # 如果这个病人提取出了有效数据，则进行体积堆叠并保存
        if len(patient_slices_rgb) > 0:
            # Stacking: list of (256, 256, 3) -> (N, 256, 256, 3)
            volume_img_arr = np.stack(patient_slices_rgb, axis=0)
            volume_mask_arr = np.stack(patient_slices_mask, axis=0)

            img_save_path = os.path.join(vol_img_dir, f"{patient_id}.npy")
            mask_save_path = os.path.join(vol_mask_dir, f"{patient_id}.npy")

            np.save(img_save_path, volume_img_arr.astype(np.float32))
            np.save(mask_save_path, volume_mask_arr.astype(np.uint8))

            # 记录到全局字典
            global_index_dict[patient_id] = {
                "img_path": f"crop_imagesTr/{patient_id}.npy",
                "mask_path": f"crop_labelsTr/{patient_id}.npy",
                "crop_params": {"y_start": int(y_start), "x_start": int(x_start)},
                "num_valid_slices": len(patient_slices_info),
                "slices_info": patient_slices_info
            }

    # ==========================================
    # 终极保存：把参数字典分别保存为 json 和 p
    # ==========================================
    json_path = os.path.join(output_dir, "MDC_annotations.json")
    pickle_path = os.path.join(output_dir, "MDC_annotations.p")

    # 1. 存 JSON (供你用肉眼 Check 文本和数据结构)
    print(f"Saving Text and Parameters to JSON: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(global_index_dict, f, indent=2, cls=NumpyEncoder)

    # 2. 存 Pickle (供 Dataloader 极速读取)
    print(f"Saving Text and Parameters to Pickle: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(global_index_dict, f)

    print("✅ 处理完成！图像已落盘，文本与参数已剥离为 JSON 和 Pickle 格式。")


if __name__ == "__main__":
    PREPROCESSED_DIR = "/root/autodl-tmp/Data/Task07_Pancreas"
    PREPROCESSED_IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_imagesTr"
    PREPROCESSED_LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_labelsTr"
    process_data_to_volumes(
        img_dir=PREPROCESSED_IMG_DIR,
        label_dir=PREPROCESSED_LBL_DIR,
        output_dir=PREPROCESSED_DIR,
        pts_pancreas=64,
        pts_tumor=32
    )
