import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_instance_areas(mask_slice, min_area=15):
    """提取单张 2D 切片中，所有独立连通域的面积"""
    # 寻找外部轮廓
    contours, _ = cv2.findContours(mask_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            areas.append(area)
    return areas

def analyze_dataset_areas(label_dir):
    """遍历整个数据集，统计胰腺和肿瘤的面积分布"""
    pancreas_areas = []
    tumor_areas = []

    # 获取所有 npy 标签文件
    npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    print(f"👀 找到 {len(npy_files)} 个病人的标签文件，开始统计...")

    for file_name in tqdm(npy_files):
        file_path = os.path.join(label_dir, file_name)
        
        # 假设你的标签 shape 是 (num_slices, H, W) 或类似格式
        # 如果是 (H, W, num_slices)，请在下面用 np.transpose 转换一下
        volume_label = np.load(file_path) 

        for slice_idx in range(volume_label.shape[0]):
            lbl_slice = volume_label[slice_idx]
            
            # 假设: 1 是正常胰腺，2 是肿瘤
            # 胰腺的整体 Mask 应该包含肿瘤部分
            panc_mask = (lbl_slice == 1) | (lbl_slice == 2)
            tumor_mask = (lbl_slice == 2)

            if np.any(panc_mask):
                panc_areas_in_slice = get_instance_areas(panc_mask)
                pancreas_areas.extend(panc_areas_in_slice)
                
            if np.any(tumor_mask):
                tumor_areas_in_slice = get_instance_areas(tumor_mask)
                tumor_areas.extend(tumor_areas_in_slice)

    return np.array(pancreas_areas), np.array(tumor_areas)

def print_statistics(name, areas):
    """打印四分位数等统计信息"""
    print(f"\n{'='*40}")
    print(f"📊 {name} 面积统计报告 (共 {len(areas)} 个独立实例)")
    print(f"{'='*40}")
    if len(areas) == 0:
        print("未找到数据！")
        return

    print(f"最小值 (Min):      {np.min(areas):.1f}")
    print(f"最大值 (Max):      {np.max(areas):.1f}")
    print(f"平均值 (Mean):     {np.mean(areas):.1f}")
    print("-" * 40)
    print("💡 推荐的分档参考点 (Percentiles):")
    print(f"25% 的实例面积小于:  {np.percentile(areas, 25):.1f}   (可作为极小/小档的分界)")
    print(f"50% 的实例面积小于:  {np.percentile(areas, 50):.1f}   (中位数，可作为小/中档的分界)")
    print(f"75% 的实例面积小于:  {np.percentile(areas, 75):.1f}   (可作为中/大档的分界)")
    print(f"90% 的实例面积小于:  {np.percentile(areas, 90):.1f}   (可作为大/超大档的分界)")
    print(f"95% 的实例面积小于:  {np.percentile(areas, 95):.1f}")

def plot_distribution(save_dir, panc_areas, tumor_areas):
    """绘制面积分布的对数直方图"""
    plt.figure(figsize=(12, 5))

    # 胰腺分布图
    plt.subplot(1, 2, 1)
    # 因为面积差异大，使用 log10 缩放 x 轴以便于观察
    plt.hist(np.log10(panc_areas + 1), bins=50, color='blue', alpha=0.7)
    plt.title('Pancreas Area Distribution (Log10)')
    plt.xlabel('Area (Log10 scale)')
    plt.ylabel('Frequency')

    # 肿瘤分布图
    plt.subplot(1, 2, 2)
    plt.hist(np.log10(tumor_areas + 1), bins=50, color='red', alpha=0.7)
    plt.title('Tumor Area Distribution (Log10)')
    plt.xlabel('Area (Log10 scale)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/MDC512_area_distribution.png')
    print("\n✅ 分布直方图已保存为 'area_distribution.png'，请查看。")


if __name__ == "__main__":
    # ⚠️ 把这里换成你存放 .npy 标签数据的文件夹路径
    SAVE_DIRECTORY = "/root/autodl-tmp/Data/Task07_Pancreas"
    print("启动面积扫描任务...")
    panc_areas, tumor_areas = analyze_dataset_areas(f'{SAVE_DIRECTORY}/crop_labelsTr_512')
    
    print_statistics("🩸 肿瘤 (Tumor)", tumor_areas)
    print_statistics("🫀 胰腺 (Pancreas)", panc_areas)
    
    plot_distribution(SAVE_DIRECTORY, panc_areas, tumor_areas)