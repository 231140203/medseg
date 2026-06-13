import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def load_and_verify_polygons(pickle_path, data_root, num_samples=10):
    # 1. 加载 .p 文件
    print(f"📦 正在加载标注文件: {pickle_path} ...")
    with open(pickle_path, 'rb') as f:
        global_index_dict = pickle.load(f)

    # 2. 将所有病人、所有切片的所有任务“打平”到一个列表里
    all_samples = []
    for patient_id, info in global_index_dict.items():
        img_path = info["img_path"]
        for slice_info in info["slices_info"]:
            all_samples.append({
                "patient_id": patient_id,
                "img_path": img_path,
                "relative_z": slice_info["relative_z"],
                "task": slice_info["task"],
                "polygons": slice_info["polygons"]
            })

    print(f"📊 总计提取到 {len(all_samples)} 个有效的 2D 任务样本。")

    # 3. 随机抽取 10 个样本
    sampled_data = random.sample(all_samples, min(num_samples, len(all_samples)))

    # 4. 准备画板 (2行5列)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    print(f"🎨 正在绘制随机抽取的 {len(sampled_data)} 个样本...")

    for i, sample in enumerate(sampled_data):
        ax = axes[i]
        
        # 加载 3D 图像卷并提取对应的 2D 切片
        full_img_path = os.path.join(data_root, sample["img_path"])
        volume_img = np.load(full_img_path, mmap_mode='r')
            
        img_slice = volume_img[sample["relative_z"]]

        # 医学图像通常需要一定的窗宽窗位调整 (1%~99%截断)，方便显示
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_slice = np.clip(img_slice, p1, p99)
        if img_slice.max() > img_slice.min():
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())

        # 显示底图
        ax.imshow(img_slice, cmap='gray')

        # 定义画笔颜色 (如果一个切片有多个独立的轮廓)
        colors = ['#FF4B4B', '#00E676', '#29B6F6', '#FFA726', '#AB47BC']

        for p_idx, poly_1d in enumerate(sample["polygons"]):
            # 将 1D 列表 [x1, y1, x2, y2...] 转换为 2D 数组 [[x1,y1], [x2,y2]...]
            poly_2d = np.array(poly_1d).reshape(-1, 2)
            color = colors[p_idx % len(colors)]

            # A. 绘制连线 (为了闭合多边形，将第一个点拼接到最后)
            closed_poly = np.vstack((poly_2d, poly_2d[0]))
            ax.plot(closed_poly[:, 0], closed_poly[:, 1], color=color, linewidth=2, zorder=2)

            # B. 绘制所有的坐标点 (小白点带彩边)
            ax.scatter(poly_2d[:, 0], poly_2d[:, 1], color=color, s=20, edgecolors='white', linewidths=0.8, zorder=3)

            # C. ⭐ 极其重要：高亮起始点 (验证左上角重排逻辑)
            ax.scatter(poly_2d[0, 0], poly_2d[0, 1], color='yellow', marker='*', s=150, edgecolors='black', zorder=4)

        # 设置标题
        title = f"ID: {sample['patient_id']} | Z: {sample['relative_z']}\nTask: {sample['task'].upper()}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout()
    
    # 保存结果
    save_path = f"{data_root}/polygon_verification_512.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化完成！图像已保存至: {os.path.abspath(save_path)}")
    
    # 如果你在带界面的环境，可以取消注释下面这行来直接弹窗查看
    # plt.show()

if __name__ == "__main__":
    # ==========================
    # 请修改为你的实际路径
    # ==========================
    DATA_ROOT = "/root/autodl-tmp/Data/Task07_Pancreas"      # 指向包含 crop_imagesTr_512 的根目录
    PICKLE_PATH = os.path.join(DATA_ROOT, "MDC512_sep_annotations.p") # 指向你最新生成的 512 尺度 .p 文件
    
    load_and_verify_polygons(PICKLE_PATH, DATA_ROOT)