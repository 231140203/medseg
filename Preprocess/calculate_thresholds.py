import os
import glob
import re
import numpy as np
import SimpleITK as sitk
from skimage import measure
from tqdm import tqdm

def extract_case_id(pid_str):
    match = re.search(r'\d+', pid_str)
    return int(match.group()) if match else 0

data_dir = "/root/autodl-tmp/processed_labelsTr"
label_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))

N_SPLITS = 5
CURRENT_FOLD = 0

pancreas_solidity = []
pancreas_eccentricity = []
tumor_solidity = []
tumor_eccentricity = []

print("正在扫描训练集标签，提取形态学物理特征...")

for file_path in tqdm(label_files):
    filename = os.path.basename(file_path)
    pid = filename.replace(".nii.gz", "")
    case_id = extract_case_id(pid)
    
    # 严格跳过验证集
    if (case_id % N_SPLITS) == CURRENT_FOLD:
        continue 
        
    label_itk = sitk.ReadImage(file_path)
    label_array = sitk.GetArrayFromImage(label_itk)
    
    for z in range(label_array.shape[0]):
        slice_label = label_array[z, :, :]
        if np.max(slice_label) == 0: continue
            
        panc_mask = ((slice_label == 1) | (slice_label == 2)).astype(int)
        tumor_mask = (slice_label == 2).astype(int)
        
        # 提取胰腺的 solidity 和 eccentricity
        if np.max(panc_mask) > 0:
            for p in measure.regionprops(measure.label(panc_mask)):
                # 过滤掉极小噪点避免干扰形态学分布
                if p.area > 10: 
                    pancreas_solidity.append(p.solidity)
                    pancreas_eccentricity.append(p.eccentricity)
                
        # 提取肿瘤的 solidity 和 eccentricity
        if np.max(tumor_mask) > 0:
            for t in measure.regionprops(measure.label(tumor_mask)):
                if t.area > 10:
                    tumor_solidity.append(t.solidity)
                    tumor_eccentricity.append(t.eccentricity)

print("\n" + "="*60)
print("🎯 训练集形态学特征阈值计算结果 (33%, 66% 分位数)")
print("="*60)

# 计算 33% 和 66% 的分界线，将数据均匀分为三份
p_sol = np.percentile(pancreas_solidity, [33, 66])
p_ecc = np.percentile(pancreas_eccentricity, [33, 66])
t_sol = np.percentile(tumor_solidity, [33, 66])
t_ecc = np.percentile(tumor_eccentricity, [33, 66])

print("【胰腺形态阈值】:")
print(f"Solidity (边缘平滑度)    : 1/3线 = {p_sol[0]:.4f}, 2/3线 = {p_sol[1]:.4f}")
print(f"Eccentricity (长宽比例) : 1/3线 = {p_ecc[0]:.4f}, 2/3线 = {p_ecc[1]:.4f}")

print("\n【肿瘤形态阈值】:")
print(f"Solidity (边缘平滑度)    : 1/3线 = {t_sol[0]:.4f}, 2/3线 = {t_sol[1]:.4f}")
print(f"Eccentricity (长宽比例) : 1/3线 = {t_ecc[0]:.4f}, 2/3线 = {t_ecc[1]:.4f}")
print("="*60)