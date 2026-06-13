import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def verify_source_data(pickle_path, img_dir, lbl_dir, save_dir, num_samples=10):
    print(f"📦 正在读取最原始的标注文件: {pickle_path} ...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # 1. 把所有切片打平，方便随机抽样
    all_slices = []
    for patient_id, info in data.items():
        # 提取文件名，例如 pancreas_001.npy
        file_name = os.path.basename(info["img_path"])
        
        for slice_info in info["slices_info"]:
            all_slices.append({
                "patient_id": patient_id,
                "file_name": file_name,
                "relative_z": slice_info["relative_z"],
                "task": slice_info["task"],
                "polygons": slice_info["polygons"]
            })

    # 随机抽取样本
    sampled_data = random.sample(all_slices, min(num_samples, len(all_slices)))

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    print("🎨 正在生成源头对比图 (原图 + 原Mask + 原始坐标)...")

    for i, sample in enumerate(sampled_data):
        ax = axes[i]
        
        img_path = os.path.join(img_dir, sample["file_name"])
        lbl_path = os.path.join(lbl_dir, sample["file_name"])

        # 2. 读取原始的 npy 图像和标签
        try:
            img_vol = np.load(img_path, mmap_mode='r')
            lbl_vol = np.load(lbl_path, mmap_mode='r')
                
            img_slice = img_vol[sample["relative_z"]]
            lbl_slice = lbl_vol[sample["relative_z"]]
        except Exception as e:
            ax.set_title(f"读取失败: {sample['file_name']}")
            continue

        # 显示原图底图 (调整窗宽窗位)
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_slice_disp = np.clip(img_slice, p1, p99)
        if img_slice_disp.max() > img_slice_disp.min():
            img_slice_disp = (img_slice_disp - img_slice_disp.min()) / (img_slice_disp.max() - img_slice_disp.min())
        ax.imshow(img_slice_disp, cmap='gray')

        # 3. 叠加真实的 Mask
        task = sample["task"]
        mask_vis = np.zeros_like(lbl_slice, dtype=float)
        alpha_vis = np.zeros_like(lbl_slice, dtype=float)
        
        # 胰腺=1, 肿瘤=2 (根据你的任务单独显示对应的 Mask)
        if task == 'pancreas':
            # 胰腺轮廓 = 正常组织(1) + 肿瘤组织(2)
            mask_condition = (lbl_slice == 1) | (lbl_slice == 2)
        else:
            # 肿瘤轮廓 = 仅仅是肿瘤(2)
            mask_condition = (lbl_slice == 2)
        
        mask_vis[mask_condition] = 1.0
        alpha_vis[mask_condition] = 0.1 # 半透明
        
        cmap = 'cool' if task == 'pancreas' else 'autumn'
        ax.imshow(mask_vis, cmap=cmap, alpha=alpha_vis, interpolation='nearest')

        # 4. 直接把 .p 文件里的点画上去 (绝对坐标，不做任何除法和变换！)
        for poly_1d in sample["polygons"]:
            poly_2d = np.array(poly_1d).reshape(-1, 2)
            
            if len(poly_2d) >= 3:
                # 闭合多边形
                closed_poly = np.vstack((poly_2d, poly_2d[0]))
                # 画出轮廓
                ax.plot(closed_poly[:, 0], closed_poly[:, 1], color='#00E676', linewidth=2)
                # 画出散点
                ax.scatter(closed_poly[:, 0], closed_poly[:, 1], color='yellow', s=15, edgecolors='black')
                # 标记起点
                ax.scatter(closed_poly[0, 0], closed_poly[0, 1], color='blue', marker='*', s=100, zorder=5)

        ax.set_title(f"ID: {sample['patient_id']} | Z: {sample['relative_z']}\nTask: {task}", fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = f"{save_dir}/source_data_verification.jpg"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 终极验尸报告已生成: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # ==========================
    # 🚨 填入你的真实目录路径 🚨
    # ==========================
    DATA_ROOT = "/root/autodl-tmp/Data/Task07_Pancreas" 
    PICKLE_PATH = os.path.join(DATA_ROOT, "MDC512_sep_annotations.p")
    
    # 指向存放 512 尺寸 (或你正在用的尺寸) .npy 的文件夹
    IMG_DIR = os.path.join(DATA_ROOT, "crop_imagesTr_512")
    LBL_DIR = os.path.join(DATA_ROOT, "crop_labelsTr_512")
    
    verify_source_data(PICKLE_PATH, IMG_DIR, LBL_DIR, DATA_ROOT)