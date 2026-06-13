import os
import glob
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader


def get_modulo_splits(data_dir, n_splits=5, fold=0):
    """
    借鉴 Dual-Task-Seg 的隔断采样法（1+4*i / 1+5*i）
    基于文件编号进行划分，确保数据分布绝对均匀
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))

    train_files = []
    val_files = []

    for f in all_files:
        basename = os.path.basename(f)
        # 提取文件名中的数字部分，例如 "pancreas_001.nii.gz" -> 1
        num_str = ''.join(filter(str.isdigit, basename))
        if not num_str:
            continue

        case_id = int(num_str)

        # 核心逻辑：利用 case_id 对 n_splits 取余来进行固定划分
        if (case_id % n_splits) == fold:
            val_files.append(f)
        else:
            train_files.append(f)

    print(f"Fold {fold}: Train samples={len(train_files)}, Val samples={len(val_files)}")
    return train_files, val_files


class PancreasZPatchDataset(Dataset):
    def __init__(self, image_files, label_dir, patch_z=8, xy_size=256,
                 is_train=True, oversample_percent=0.5, train_samples_per_image=10):
        """
        patch_z: Z 轴最大张数 (8)
        xy_size: XY 平面大小 (256)
        train_samples_per_image: 训练时，一个病人图像被重复采样的次数
        """
        self.image_files = image_files
        self.label_dir = label_dir
        self.patch_z = patch_z
        self.xy_size = xy_size
        self.is_train = is_train
        self.oversample_percent = oversample_percent
        self.train_samples_per_image = train_samples_per_image

    def __len__(self):
        if self.is_train:
            # 解决 Z 轴太小导致样本不够的问题：把数据集长度放大
            # 例如 200 个训练病人，每个病人切 10 次，epoch 长度即为 2000
            return len(self.image_files) * self.train_samples_per_image
        else:
            return len(self.image_files)

    def pad_if_needed(self, img, lbl):
        """Z 轴如果小于 8 张，或者 XY 小于 256，进行边界 Padding"""
        d, h, w = img.shape
        pad_d = max(0, self.patch_z - d)
        pad_h = max(0, self.xy_size - h)
        pad_w = max(0, self.xy_size - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # 仅在末尾 pad (0, pad_size)
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
            img = np.pad(img, pad_width, mode='constant', constant_values=0)
            lbl = np.pad(lbl, pad_width, mode='constant', constant_values=0)
        return img, lbl

    def get_random_z_slice(self, d, lbl):
        """训练时：沿 Z 轴随机切块，并结合前景过采样"""
        # 1. 前景过采样 (胰腺或肿瘤)
        if random.random() < self.oversample_percent and np.any(lbl > 0):
            foreground_z_indices = np.unique(np.where(lbl > 0)[0])
            # 随机挑一个包含前景的 Z 轴切片作为中心
            center_z = random.choice(foreground_z_indices)
            z_start = max(0, min(center_z - self.patch_z // 2, d - self.patch_z))
        else:
            # 2. 纯随机
            z_start = random.randint(0, d - self.patch_z)

        return slice(z_start, z_start + self.patch_z)

    def __getitem__(self, index):
        # 训练模式下，还原真实的病人文件索引
        if self.is_train:
            file_idx = index // self.train_samples_per_image
        else:
            file_idx = index

        img_path = self.image_files[file_idx]
        filename = os.path.basename(img_path)
        lbl_path = os.path.join(self.label_dir, filename)

        img_npy = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        lbl_npy = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))

        img_npy, lbl_npy = self.pad_if_needed(img_npy, lbl_npy)
        d, h, w = img_npy.shape

        # 假设之前的预处理已经将 XY 切/缩放到了 256
        # 为了严谨，在此强制取中心 256x256
        y_start = (h - self.xy_size) // 2
        x_start = (w - self.xy_size) // 2
        xy_slicer = (slice(y_start, y_start + self.xy_size), slice(x_start, x_start + self.xy_size))

        # ================== 训练时：返回单个 (1, 8, 256, 256) ==================
        if self.is_train:
            z_slicer = self.get_random_z_slice(d, lbl_npy)

            img_patch = img_npy[z_slicer, xy_slicer[0], xy_slicer[1]]
            lbl_patch = lbl_npy[z_slicer, xy_slicer[0], xy_slicer[1]]

            # 转为 Tensor (C, Z, Y, X)
            return (
                torch.from_numpy(img_patch[None, ...]).float(),
                torch.from_numpy(lbl_patch[None, ...]).long()
            )

        # ================== 验证时：沿 Z 轴滑动，返回所有 Chunk ==================
        else:
            # 步长 (Stride) 可以是 patch_z (无重叠) 或更小 (有重叠)
            # 这里默认无重叠切分，方便拼接
            stride = self.patch_z

            img_chunks = []
            z_starts = []

            for z in range(0, d, stride):
                z_start = z
                z_end = z_start + self.patch_z

                # 处理末尾不足 8 张的情况：往前拉回凑满 8 张
                if z_end > d:
                    z_start = max(0, d - self.patch_z)
                    z_end = d

                chunk = img_npy[z_start:z_end, xy_slicer[0], xy_slicer[1]]
                img_chunks.append(chunk[None, ...])  # 加入 Channel 维度
                z_starts.append(z_start)

            # 将该病人的所有块叠在一起，形状变为 (N_chunks, 1, 8, 256, 256)
            img_chunks_tensor = torch.from_numpy(np.stack(img_chunks)).float()

            # 把完整的原始标签一起返回，用于拼合后算 Dice
            lbl_tensor = torch.from_numpy(lbl_npy).long()

            return img_chunks_tensor, z_starts, lbl_tensor, d