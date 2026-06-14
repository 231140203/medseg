#!/usr/bin/env python3
"""
单病人推理脚本 — CPU 验证全链路
从 NIfTI → 预处理 → 模型推理 → 3D 重建 → 保存 nii.gz

Usage:
  cd /root/autodl-tmp/MRIpolySeg
  python3 run_inference.py --patient pancreas_001 --num-slices 1
"""

import os, sys, argparse, math, json
import numpy as np

# === NumPy 兼容补丁 ===
np.float = float; np.int = int; np.complex = complex; np.bool = bool

# === 路径设置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fairseq'))

import torch
import nibabel as nib
import cv2
from tqdm import tqdm

from fairseq import utils
from fairseq.tasks import TASK_REGISTRY
from fairseq.data import Dictionary

from tasks.MDC_pretrain import MDCPretrainTask
from data.MDC_dataset import MedicalPolyDataset

# 触发模型注册
from models.polyformer import polyformer  # noqa: F401

# ============================================================
# Config
# ============================================================
CKPT_PATH = "/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt"
IMG_DIR = "/root/autodl-tmp/processed_imagesTr"
LBL_DIR = "/root/autodl-tmp/processed_labelsTr"
OUTPUT_DIR = "/root/autodl-tmp/MRIpolySeg/Results/inference_test"
DATA_ROOT = "/root/autodl-tmp/Data/Task07_Pancreas"
PICKLE_PATH = "/root/autodl-tmp/MRIpolySeg/data/MDC512_new_annotations.p"

PATCH_SIZE = 256
NUM_BINS = 64
MAX_LEN = 400
MIN_LEN = 40


def load_model_and_task():
    """加载训练好的模型 + Task"""
    print(f"\n[1/4] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False, mmap=True)

    cfg_dict = ckpt['cfg']
    # task 和 model 的 cfg 在同一个 Namespace 里
    task_cfg = {k: v for k, v in cfg_dict.items()}

    # 确保关键路径正确
    task_cfg['data_root'] = DATA_ROOT
    task_cfg['pickle_path'] = PICKLE_PATH

    # 构建词典 — 必须与训练时完全一致
    # 训练时 vocab = 4 特殊 token (bos=0, pad=1, eos=2, unk=3) + 64×64 bin tokens = 4100
    # BPE dict.txt 绝不能加载到 embedding 词典中！文本 tokenization 由 BERT 另行处理。
    src_dict = Dictionary()  # 已有 bos, pad, eos, unk (4 tokens)
    tgt_dict = Dictionary()
    # 只添加 bin tokens，不加载 BPE
    for i in range(NUM_BINS):
        for j in range(NUM_BINS):
            src_dict.add_symbol(f"<bin_{i}_{j}>")
            tgt_dict.add_symbol(f"<bin_{i}_{j}>")

    print(f"  Source dict: {len(src_dict)} types")
    print(f"  Target dict: {len(tgt_dict)} types")

    # 构建 Task
    from argparse import Namespace
    task_cfg_ns = Namespace(**task_cfg)
    task = MDCPretrainTask(task_cfg_ns, src_dict, tgt_dict)

    # 构建模型
    from models.polyformer.polyformer import PolyFormerModel
    from fairseq.models import ARCH_MODEL_REGISTRY

    model_cfg = cfg_dict['model'] if isinstance(cfg_dict['model'], Namespace) else Namespace(**cfg_dict['model'])
    model_arch = getattr(model_cfg, 'arch', 'polyformer_b')
    print(f"  Model arch: {model_arch}")
    print(f"  ARCH_MODEL_REGISTRY has {model_arch}: {model_arch in ARCH_MODEL_REGISTRY}")

    # 确保 task 有必要的属性
    if not hasattr(task, 'cfg'):
        task.cfg = task_cfg_ns

    # 使用 ARCH_MODEL_REGISTRY（legacy model）来构建模型
    model_cls = ARCH_MODEL_REGISTRY[model_arch]
    model = model_cls.build_model(model_cfg, task)

    # 加载权重
    model_state = ckpt.get('model', ckpt)
    model.load_state_dict(model_state, strict=False)

    model.eval()
    device = torch.device('cuda')
    model = model.to(device).float()  # fp32 on GPU for reliability

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model arch: {model_arch}")
    print(f"  Params: {n_params/1e6:.1f}M")
    print(f"  Device: cuda (fp32)")

    return model, task


def preprocess_3d_volume(nii_path, target_size=256):
    """与 create_MDC_dataset.py 完全一致的预处理：
    1. MinMax normalize [0, 1]
    2. fliplr(rot90(data, k=1))  (X, Y, Z) → (H, W, Z)
    3. Smart crop 256×256 (前景中心)

    Returns: img_3d (Z, H, W, 3), crop_params (y_start, x_start), original_header
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    header = nii.header

    # Step 1: MinMax normalize [0, 1]
    min_v, max_v = data.min(), data.max()
    if max_v - min_v > 1e-6:
        data = (data - min_v) / (max_v - min_v)
    else:
        data = np.zeros_like(data)

    # Step 2: rot90 + fliplr (与 create_MDC_dataset.py 一致)
    # nii 加载后 shape = (X, Y, Z)，rot90(k=1) 在 XY 平面旋转，fliplr 左右翻转
    data = np.fliplr(np.rot90(data, k=1))  # (H', W', Z)

    h, w, d = data.shape

    # Step 3: Smart Crop — 前景中心裁剪到 target_size×target_size
    fg_mask = data > 0.01
    xy_proj = np.any(fg_mask, axis=2)

    if not np.any(xy_proj):
        y_start = h // 2 - target_size // 2
        x_start = w // 2 - target_size // 2
    else:
        y_idx, x_idx = np.where(xy_proj)
        min_y, max_y = np.min(y_idx), np.max(y_idx)
        min_x, max_x = np.min(x_idx), np.max(x_idx)
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        y_start = center_y - target_size // 2
        x_start = center_x - target_size // 2
        y_start = max(0, min(y_start, h - target_size))
        x_start = max(0, min(x_start, w - target_size))

    # 预分配输出 (Z, H, W, 3)
    img_4d = np.zeros((d, target_size, target_size, 3), dtype=np.float32)
    for z in range(d):
        slc = data[y_start:y_start+target_size, x_start:x_start+target_size, z]
        # 如果裁剪区域小于 target (边界情况), resize
        if slc.shape[0] != target_size or slc.shape[1] != target_size:
            slc = cv2.resize(slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        # 3通道灰度
        slc = np.clip(slc, 0, 1)
        for c in range(3):
            img_4d[z, :, :, c] = slc

    return img_4d, (y_start, x_start), header


def run_inference_single_slice(model, task, img_rgb, prompt_text="segment a pancreas"):
    """对单张切片运行自回归推理，返回 pred_mask (H, W)"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    h, w = img_rgb.shape[:2]

    # 图像 Tensor: (1, 3, H, W)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device=device, dtype=dtype)

    # BERT Tokenize — 直接从 pretrained 路径加载，不依赖 task.datasets
    from bert.tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/pretrained_weights/RadBERT/')

    full_prompt = f' which region does the text " {prompt_text} " describe?'
    tokens = tokenizer(full_prompt, return_tensors='pt', padding=True, truncation=True, max_length=80)
    src_tokens = tokens['input_ids'].to(device)
    src_lengths = torch.tensor([src_tokens.shape[1]]).to(device)
    att_masks = tokens['attention_mask'].to(device)

    # 构建 sample dict
    patch_mask_tensor = torch.ones(1, dtype=torch.bool)
    sample = {
        'id': np.array(['single_slice']),
        'net_input': {
            'patch_images': img_tensor,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'att_masks': att_masks,
            'patch_masks': patch_mask_tensor,
        },
        'label': [np.zeros((h, w), dtype=np.uint8)],  # 推理时无 GT, 只用于 shape
        'task': ['pancreas'],  # 推理任务类型
    }

    # 调用 task 的推理方法
    try:
        pred_masks, _ = task.get_predictions_and_masks(model, sample)
        pred_mask = pred_masks[0]  # (H, W) uint8
    except Exception as e:
        print(f"  ERROR in inference: {e}")
        import traceback; traceback.print_exc()
        pred_mask = np.zeros((h, w), dtype=np.uint8)

    return pred_mask


def reverse_transform(vol):
    """反向变换: fliplr(np.rot90(data, k=1)) → 还原
    vol: (Z, H, W)
    """
    out = []
    for i in range(vol.shape[0]):
        slc = vol[i]
        slc = np.rot90(slc, k=3)   # 反 rot90(k=1)
        slc = np.fliplr(slc)       # 反 fliplr
        out.append(slc)
    return np.stack(out, axis=0)


def run_inference_on_patient(model, task, img_path, lbl_path, num_slices=None, start_slice=0):
    """对单个病人所有切片推理，返回 pred_3d_original, gt_3d_original, header"""
    patient_id = os.path.basename(img_path).replace('.nii.gz', '')

    # 加载原始图像信息
    orig_nii = nib.load(img_path)
    orig_shape = orig_nii.shape
    orig_header = orig_nii.header

    # 预处理图像
    print(f"  Preprocessing {patient_id} ...")
    img_4d, (y_start, x_start), _ = preprocess_3d_volume(img_path, PATCH_SIZE)
    z_dim = img_4d.shape[0]

    # 预处理标签（同样变换）
    lbl_nii = nib.load(lbl_path)
    lbl_data = lbl_nii.get_fdata().astype(np.float32)
    lbl_data = np.fliplr(np.rot90(lbl_data, k=1))
    h_full, w_full, d_full = lbl_data.shape

    # 限制切片数 — 从 start_slice 开始
    slice_start = max(0, min(start_slice, z_dim - 1))
    slice_end = z_dim if num_slices is None else min(z_dim, slice_start + num_slices)
    z_indices = list(range(slice_start, slice_end))

    print(f"  Total slices: {len(z_indices)} (z={slice_start}..{slice_end-1}), crop=({y_start},{x_start})")

    pred_slices = []
    gt_slices = []
    img_slices = []  # 保存预处理后的图像用于可视化

    for z in tqdm(z_indices, desc=f"  {patient_id}"):
        img_rgb = img_4d[z]

        # 推理
        pred_mask = run_inference_single_slice(model, task, img_rgb)
        pred_slices.append(pred_mask)

        # 保存预处理图像（取灰度通道，shape (256,256)）
        img_slices.append((img_rgb[..., 0] * 255).astype(np.uint8))

        # GT 裁剪
        gt_slc = lbl_data[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE, z]
        if gt_slc.shape[0] != PATCH_SIZE or gt_slc.shape[1] != PATCH_SIZE:
            gt_slc = cv2.resize(gt_slc.astype(np.float32), (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        gt_slices.append(gt_slc.astype(np.uint8))

    # Stack → 3D
    pred_3d_crop = np.stack(pred_slices, axis=0)  # (Z, 256, 256)
    gt_3d_crop = np.stack(gt_slices, axis=0)
    img_3d_crop = np.stack(img_slices, axis=0)

    # 反裁剪：放回旋转+翻转后的全尺寸空间
    pred_full = np.zeros((d_full, h_full, w_full), dtype=np.uint8)
    gt_full = np.zeros((d_full, h_full, w_full), dtype=np.uint8)
    img_full = np.zeros((d_full, h_full, w_full), dtype=np.uint8)

    for z in range(min(d_full, pred_3d_crop.shape[0])):
        h_avail = min(PATCH_SIZE, h_full - y_start)
        w_avail = min(PATCH_SIZE, w_full - x_start)
        pred_full[z, y_start:y_start+h_avail, x_start:x_start+w_avail] = \
            pred_3d_crop[z, :h_avail, :w_avail]
        gt_full[z, y_start:y_start+h_avail, x_start:x_start+w_avail] = \
            gt_3d_crop[z, :h_avail, :w_avail]
        img_full[z, y_start:y_start+h_avail, x_start:x_start+w_avail] = \
            img_3d_crop[z, :h_avail, :w_avail]

    # 反向旋转
    pred_reversed = reverse_transform(pred_full)
    gt_reversed = reverse_transform(gt_full)
    img_reversed = reverse_transform(img_full)

    # 转回原始方向 (X, Y, Z)
    # 原始: (X,Y,Z) → rot90+fliplr → (H,W,Z)
    # 我们需要从 (Z, H_rev, W_rev) → (X, Y, Z) 用原始header保存
    return pred_reversed, gt_reversed, img_reversed, orig_header, orig_shape


def compute_dice(pred, gt, class_val):
    """计算单个类别的 Dice score"""
    p = (pred == class_val).astype(np.float32)
    g = (gt == class_val).astype(np.float32)
    intersection = (p * g).sum()
    if p.sum() + g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    return float(2.0 * intersection / (p.sum() + g.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=str, default='pancreas_001')
    parser.add_argument('--num-slices', type=int, default=None,
                        help='最多推理的切片数（None=全部）')
    parser.add_argument('--start-slice', type=int, default=0,
                        help='从第几层开始推理（0-based, 默认0）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--prompt', type=str, default='segment a pancreas')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  MRIpolySeg — 单病人推理")
    print(f"  Patient:  {args.patient}")
    print(f"  Slices:   {args.num_slices or 'all'} (start={args.start_slice})")
    print(f"  Device:   CUDA (RTX 3090)")
    print(f"  Output:   {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    # 1. 加载模型
    model, task = load_model_and_task()

    # 2. 推理
    img_path = os.path.join(IMG_DIR, f"{args.patient}.nii.gz")
    lbl_path = os.path.join(LBL_DIR, f"{args.patient}.nii.gz")

    if not os.path.exists(img_path):
        print(f"[SKIP] Image not found: {img_path}")
        return
    if not os.path.exists(lbl_path):
        print(f"[SKIP] Label not found: {lbl_path}")
        return

    print(f"\n[2/4] Running inference on {args.patient}...")
    pred_3d, gt_3d, img_3d, header, orig_shape = run_inference_on_patient(
        model, task, img_path, lbl_path, num_slices=args.num_slices, start_slice=args.start_slice
    )

    # 3. 保存 nii.gz
    print(f"\n[3/4] Saving results...")

    # pred 和 gt 在 reverse_transform 后是 (Z, Y, X) 形状，需要重塑到原始 (X, Y, Z)
    # reverse_transform 输出 (Z, H_rev, W_rev)，我们需要 transponse 回 (X, Y, Z)
    pred_nii_data = pred_3d.transpose(2, 1, 0)  # (Z, Y, X) → (X, Y, Z)
    gt_nii_data = gt_3d.transpose(2, 1, 0)
    img_nii_data = img_3d.transpose(2, 1, 0)

    pred_path = os.path.join(args.output, f"{args.patient}_pred.nii.gz")
    gt_path = os.path.join(args.output, f"{args.patient}_gt.nii.gz")
    img_path_out = os.path.join(args.output, f"{args.patient}_img.nii.gz")

    pred_nii = nib.Nifti1Image(pred_nii_data.astype(np.float32), None, header)
    nib.save(pred_nii, pred_path)
    print(f"  Saved: {pred_path}  shape={pred_nii_data.shape}")

    gt_nii = nib.Nifti1Image(gt_nii_data.astype(np.float32), None, header)
    nib.save(gt_nii, gt_path)
    print(f"  Saved: {gt_path}  shape={gt_nii_data.shape}")

    img_nii = nib.Nifti1Image(img_nii_data.astype(np.uint8), None, header)
    nib.save(img_nii, img_path_out)
    print(f"  Saved: {img_path_out}  shape={img_nii_data.shape}")

    # 4. Dice 评估
    print(f"\n[4/4] Evaluation")
    print(f"  {'-'*40}")
    print(f"  Class        Dice")
    print(f"  {'-'*40}")

    for cls_val, cls_name in [(1, 'Pancreas'), (2, 'Tumor')]:
        dice = compute_dice(pred_3d, gt_3d, cls_val)
        print(f"  {cls_name:<12} {dice:.4f}")

    print(f"  {'-'*40}")
    print(f"\n  Done! Output: {args.output}")


if __name__ == '__main__':
    main()
