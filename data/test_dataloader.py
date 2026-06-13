import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fairseq.data import Dictionary
import re
# ==========================================
# 🚨 1. 导入你的实际 Dataset (请根据你的项目路径修改)
# ==========================================
# 例如: from data.MDC_dataset import MDCDataset 
from MDC_dataset import MedicalPolyDataset

def denormalize_image(tensor_img):
    img = tensor_img.clone().detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    return np.clip(img, 0, 1)


def test_dataloader(data_root, pickle_path, bpe_dir, patch_size=512, num_bins=64):
    print("🚀 正在加载预训练词表 (Dictionary)...")
    dict_path = os.path.join(bpe_dir, "dict.txt")
    src_dict = Dictionary.load(dict_path)
    pad_idx = src_dict.pad()  # 获取占位符的 ID

    print("🚀 正在初始化 Dataset...")
    dataset = MedicalPolyDataset(
        data_root=data_root,
        pickle_path=pickle_path,
        split='train',
        src_dict=src_dict,
        tgt_dict=src_dict,
        max_src_length=80,
        max_image_size=patch_size,
        num_bins=num_bins,
        bpe=None
    )

    dataloader = DataLoader(
        dataset, batch_size=10, shuffle=True,
        collate_fn=dataset.collater, num_workers=0
    )

    batch = next(iter(dataloader))

    ids = batch.get('id', [])
    tasks = batch.get('task', ['unknown'] * 10)
    texts = batch.get('text', ['unknown'] * 10)
    patch_images = batch['net_input']['patch_images']
    targets = batch['target']  # 里面全都是词表数字 ID
    patch_masks = batch['label']

    print("\n" + "=" * 60)
    print("📋 [DataLoader 真实对齐核对报告]")
    print("=" * 60)

    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i in range(len(ids)):
        ax = axes[i]

        # 1. 直接拿到 (N, 2) 的浮点数坐标张量
        raw_coords = targets[i].cpu().numpy()

        # 提取真实坐标（过滤掉 padding 和特殊符号）
        poly_data = raw_coords[2:]
        polygons = []
        current_poly = []

        for pt in poly_data:
            # 过滤掉 padding 占位符 (通常是 1.0 或 0.0)
            if np.isclose(pt[0], 1.0) and np.isclose(pt[1], 1.0) or pt[0] == pad_idx:
                continue

            # 🚨 遇到 [0, 0] 意味着当前多边形结束，下一个多边形开始
            if np.isclose(pt[0], 0.0) and np.isclose(pt[1], 0.0):
                if len(current_poly) >= 3:
                    polygons.append(np.array(current_poly))
                current_poly = []
            else:
                current_poly.append(pt)
        valid_length = len(poly_data)
        print(f"[{i:02d}] ID: {ids[i]} | Task: {tasks[i].upper()} | 有效坐标点数: {valid_length}")

        # 显示图像
        img_vis = denormalize_image(patch_images[i])
        ax.imshow(img_vis)

        mask_vis = patch_masks[i]
        print(mask_vis.shape, patch_masks.shape)
        # 处理维度，确保最终是 2D 的 (H, W)
        if mask_vis.ndim == 3:
            if mask_vis.shape[0] == 1:
                mask_vis = mask_vis[0]
            elif mask_vis.shape[-1] == 1:
                mask_vis = mask_vis[:, :, 0]

        # 创建透明度矩阵：背景完全透明 (0)，前景半透明 (0.4)
        alpha_channel = np.where(mask_vis > 0, 0.4, 0.0)

        # 叠加 Mask (使用 cool 或 autumn 这种亮色调与原图形成对比)
        ax.imshow(mask_vis, cmap='cool', alpha=alpha_channel, interpolation='nearest')
        # ==========================================

        # 别漏了最后一个多边形
        if len(current_poly) >= 3:
            polygons.append(np.array(current_poly))

        # 3. 独立绘制每个多边形
        color = '#00E676' if tasks[i].upper() == 'PANCREAS' else '#FF4B4B'
        for poly_pts in polygons:
            poly_pts = poly_pts * patch_size  # 换算回 512 像素
            pts_closed = np.vstack((poly_pts, poly_pts[0]))

            # 使用虚线画多边形，不连贯的地方就是断开的
            ax.plot(pts_closed[:, 0], pts_closed[:, 1], color=color, linewidth=2, linestyle='--')
            ax.scatter(pts_closed[:, 0], pts_closed[:, 1], c='yellow', s=10, edgecolors='black', zorder=4)
            # 标出起点
            ax.scatter(pts_closed[0, 0], pts_closed[0, 1], c='blue', marker='*', s=80, zorder=5)
        title = f"Task: {tasks[i].upper()}\nPoints: {valid_length}"
        ax.set_title(title, fontsize=11, fontweight='bold', color='blue' if tasks[i].upper() == 'PANCREAS' else 'red')
        ax.axis('off')

    plt.tight_layout()
    save_path = f"{DATA_ROOT}/dataloader_verification_with_polygons.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 终极验证报告已保存至: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    # 请根据实际路径修改
    DATA_ROOT = "/root/autodl-tmp/Data/Task07_Pancreas"
    PICKLE_PATH = os.path.join(DATA_ROOT, "MDC512_sep_annotations.p")
    BPE_DIR = "/root/autodl-tmp/MRIPolySeg/utils/BPE"

    # 保持与你训练脚本一致的参数
    test_dataloader(DATA_ROOT, PICKLE_PATH, BPE_DIR, patch_size=512, num_bins=64)
