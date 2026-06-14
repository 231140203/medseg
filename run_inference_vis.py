#!/usr/bin/env python3
"""
全病人推理 + 逐切片 Dice + Top/Bottom 可视化
Usage: python3 run_inference_vis.py --patient pancreas_001
"""

import os, sys, argparse
import numpy as np

np.float = float; np.int = int; np.complex = complex; np.bool = bool

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fairseq'))

import torch, nibabel, cv2
from tqdm import tqdm
from argparse import Namespace
from fairseq.data import Dictionary
from tasks.MDC_pretrain import MDCPretrainTask
from models.polyformer import polyformer  # noqa
from bert.tokenization_bert import BertTokenizer

CKPT_PATH  = "/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt"
IMG_DIR    = "/root/autodl-tmp/processed_imagesTr"
LBL_DIR    = "/root/autodl-tmp/processed_labelsTr"
OUTPUT_DIR = "/root/autodl-tmp/MRIpolySeg/Results/inference_test"
DATA_ROOT  = "/root/autodl-tmp/Data/Task07_Pancreas"
PICKLE_PATH = "/root/autodl-tmp/MRIpolySeg/data/MDC512_new_annotations.p"
PATCH_SIZE = 256
NUM_BINS   = 64


def load_model_and_task():
    print(f"\n[1/3] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False, mmap=True)
    cfg_dict = ckpt['cfg']
    task_cfg = {k: v for k, v in cfg_dict.items()}
    task_cfg['data_root'] = DATA_ROOT
    task_cfg['pickle_path'] = PICKLE_PATH

    src_dict = Dictionary()
    tgt_dict = Dictionary()
    for i in range(NUM_BINS):
        for j in range(NUM_BINS):
            src_dict.add_symbol(f"<bin_{i}_{j}>")
            tgt_dict.add_symbol(f"<bin_{i}_{j}>")

    task = MDCPretrainTask(Namespace(**task_cfg), src_dict, tgt_dict)

    from fairseq.models import ARCH_MODEL_REGISTRY
    model_cfg = cfg_dict['model'] if isinstance(cfg_dict['model'], Namespace) else Namespace(**cfg_dict['model'])
    model_arch = getattr(model_cfg, 'arch', 'polyformer_b')
    model_cls = ARCH_MODEL_REGISTRY[model_arch]
    model = model_cls.build_model(model_cfg, task)

    model_state = ckpt.get('model', ckpt)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    device = torch.device('cuda')
    model = model.to(device).float()
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, device=cuda")
    return model, task


def preprocess_3d_volume(nii_path, target_size=256):
    nii = nibabel.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    header = nii.header
    min_v, max_v = data.min(), data.max()
    if max_v - min_v > 1e-6:
        data = (data - min_v) / (max_v - min_v)
    else:
        data = np.zeros_like(data)
    data = np.fliplr(np.rot90(data, k=1))
    h, w, d = data.shape
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
    img_4d = np.zeros((d, target_size, target_size, 3), dtype=np.float32)
    for z in range(d):
        slc = data[y_start:y_start+target_size, x_start:x_start+target_size, z]
        if slc.shape[0] != target_size or slc.shape[1] != target_size:
            slc = cv2.resize(slc, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        slc = np.clip(slc, 0, 1)
        for c in range(3):
            img_4d[z, :, :, c] = slc
    return img_4d, (y_start, x_start), header


def run_single_slice(model, task, img_rgb, tokenizer, prompt_text="segment a pancreas"):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h, w = img_rgb.shape[:2]

    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device=device, dtype=dtype)

    full_prompt = f' which region does the text " {prompt_text} " describe?'
    tokens = tokenizer(full_prompt, return_tensors='pt', padding=True, truncation=True, max_length=80)
    src_tokens = tokens['input_ids'].to(device)
    src_lengths = torch.tensor([src_tokens.shape[1]]).to(device)
    att_masks = tokens['attention_mask'].to(device)
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
        'label': [np.zeros((h, w), dtype=np.uint8)],
        'task': ['pancreas'],
    }

    try:
        pred_masks, _ = task.get_predictions_and_masks(model, sample)
        pred_mask = pred_masks[0]
    except Exception as e:
        print(f"  ERROR at slice: {e}")
        pred_mask = np.zeros((h, w), dtype=np.uint8)
    return pred_mask


def dice_score(pred, gt, class_val):
    p = (pred == class_val).astype(np.float32)
    g = (gt == class_val).astype(np.float32)
    inter = (p * g).sum()
    if inter == 0:
        return 0.0
    return float(2.0 * inter / (p.sum() + g.sum()))


def make_vis_panel(img_slc, gt_slc, pred_slc, z_idx, d_panc, d_tumor):
    """创建可视化面板: 原图|GT|预测|叠加 1280x256"""
    # 归一化图像到 0-255
    img_v = np.clip(img_slc, 0, 1)
    img_u8 = (img_v * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    # GT overlay
    gt_overlay = img_color.copy()
    gt_overlay[gt_slc == 1] = [255, 0, 0]   # 红=胰腺
    gt_overlay[gt_slc == 2] = [0, 255, 0]   # 绿=肿瘤

    # Pred overlay
    pred_overlay = img_color.copy()
    pred_overlay[pred_slc == 1] = [0, 0, 255]  # 红=预测胰腺
    pred_overlay[pred_slc == 2] = [0, 255, 0]  # 绿=预测肿瘤

    # GT mask (纯色)
    gt_vis = np.zeros_like(img_color)
    gt_vis[gt_slc == 1] = [255, 255, 255]
    gt_vis[gt_slc == 2] = [128, 128, 128]

    # Pred mask (纯色)
    pred_vis = np.zeros_like(img_color)
    pred_vis[pred_slc == 1] = [255, 255, 255]
    pred_vis[pred_slc == 2] = [128, 128, 128]

    # 拼成一行: 原图 | GTmask | Predmask | GT叠加 | Pred叠加
    panel = np.hstack([img_color, gt_vis, pred_vis, gt_overlay, pred_overlay])

    # 加文字
    info = f'z={z_idx:03d} | Pancreas Dice={d_panc:.4f} Tumor Dice={d_tumor:.4f}'
    cv2.putText(panel, info, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 每栏标题
    for i, label in enumerate(['Image', 'GT', 'Pred', 'GT+Img', 'Pred+Img']):
        x0 = i * 256 + 5
        cv2.putText(panel, label, (x0, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=str, default='pancreas_001')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--prompt', type=str, default='segment a pancreas')
    args = parser.parse_args()

    patient_id = args.patient
    vis_dir = os.path.join(args.output, f'{patient_id}_slices')
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Full patient inference + per-slice Dice + Top/Bottom vis")
    print(f"  Patient:  {patient_id}")
    print("=" * 60)

    # 1. 加载模型
    model, task = load_model_and_task()
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/pretrained_weights/RadBERT/')

    # 2. 加载&预处理数据
    img_path = os.path.join(IMG_DIR, f"{patient_id}.nii.gz")
    lbl_path = os.path.join(LBL_DIR, f"{patient_id}.nii.gz")
    print(f"\n[2/3] Preprocessing {patient_id}...")
    img_4d, (y_start, x_start), header = preprocess_3d_volume(img_path, PATCH_SIZE)
    z_dim = img_4d.shape[0]

    lbl_nii = nibabel.load(lbl_path)
    lbl_data = lbl_nii.get_fdata().astype(np.float32)
    lbl_data = np.fliplr(np.rot90(lbl_data, k=1))

    print(f"  Slices: {z_dim}, crop=({y_start},{x_start})")

    # 3. 逐切片推理 + 记录 Dice
    print(f"\n[3/3] Inference...")
    records = []  # [(z, img_slc, gt_slc, pred_slc, dice_panc, dice_tumor)]

    for z in tqdm(range(z_dim), desc=f"  {patient_id}"):
        img_rgb = img_4d[z]

        # 推理
        pred_mask = run_single_slice(model, task, img_rgb, tokenizer, args.prompt)

        # GT 裁剪
        gt_slc = lbl_data[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE, z]
        if gt_slc.shape[0] != PATCH_SIZE or gt_slc.shape[1] != PATCH_SIZE:
            gt_slc = cv2.resize(gt_slc.astype(np.float32), (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        gt_slc = gt_slc.astype(np.uint8)

        # 图像灰度
        img_slc = img_rgb[..., 0]

        # 逐切片 Dice
        d_panc = dice_score(pred_mask, gt_slc, 1)
        d_tumor = dice_score(pred_mask, gt_slc, 2)
        records.append((z, img_slc, gt_slc, pred_mask, d_panc, d_tumor))

    # 4. 排序 & 可视化
    # 胰腺 Dice 排序
    records_by_panc = sorted(records, key=lambda r: r[4])  # 按 pancreas dice 升序

    print(f"\n{'='*60}")
    print(f"  Per-slice Dice Summary")
    print(f"{'='*60}")

    # 统计
    panc_scores = [r[4] for r in records]
    tumor_scores = [r[5] for r in records]
    n_panc_nz = sum(1 for r in records if r[2].max() > 0)  # GT 非零切片

    print(f"  Slices with GT > 0: {n_panc_nz} / {z_dim}")
    print(f"  Pancreas Dice (on GT>0 slices):")
    nz_panc = [r[4] for r in records if r[2].max() > 0]
    if nz_panc:
        print(f"    Mean:  {np.mean(nz_panc):.4f}")
        print(f"    Median: {np.median(nz_panc):.4f}")
        print(f"    Max:    {np.max(nz_panc):.4f}")
        print(f"    Min:    {np.min(nz_panc):.4f}")
    nz_tumor = [r[5] for r in records if (r[2] == 2).sum() > 0]
    if nz_tumor:
        print(f"  Tumor Dice (on GT>0 slices where tumor exists):")
        print(f"    Mean:  {np.mean(nz_tumor):.4f}")
        print(f"    Median: {np.median(nz_tumor):.4f}")
    else:
        print(f"  Tumor:  N/A (no tumor in GT)")

    # 全局 Dice (整个 3D volume)
    all_pred = np.stack([r[3] for r in records], axis=0)
    all_gt = np.stack([r[2] for r in records], axis=0)
    global_panc = dice_score(all_pred, all_gt, 1)
    global_tumor = dice_score(all_pred, all_gt, 2)
    print(f"\n  Global 3D Pancreas Dice: {global_panc:.4f}")
    print(f"  Global 3D Tumor Dice:    {global_tumor:.4f}")

    # 5. 保存 Top 10 & Bottom 10 (按胰腺 Dice)
    print(f"\n  Saving visualization panels...")

    # 只取 GT 非零的切片排 Dice
    nz_records = [r for r in records if r[2].max() > 0]
    nz_by_panc = sorted(nz_records, key=lambda r: r[4])

    top10 = nz_by_panc[-10:]   # Dice 最高
    bot10 = nz_by_panc[:10]    # Dice 最低

    for tag, recs in [('best', top10), ('worst', bot10)]:
        for rank, (z, img_slc, gt_slc, pred_slc, d_panc, d_tumor) in enumerate(recs):
            panel = make_vis_panel(img_slc, gt_slc, pred_slc, z, d_panc, d_tumor)
            fname = f'{patient_id}_{tag}_{rank:02d}_z{z:03d}_d{d_panc:.3f}.png'
            cv2.imwrite(os.path.join(vis_dir, fname), panel)

    # 6. 也保存全部切片 Dice 的 CSV
    csv_path = os.path.join(vis_dir, f'{patient_id}_per_slice_dice.csv')
    with open(csv_path, 'w') as f:
        f.write('z,gt_has_pancreas,gt_has_tumor,dice_pancreas,dice_tumor\n')
        for z, img_slc, gt_slc, pred_slc, d_panc, d_tumor in records:
            f.write(f'{z},{int(gt_slc.max()>0)},{(gt_slc==2).sum()>0},{d_panc:.6f},{d_tumor:.6f}\n')

    print(f"\n  Done!")
    print(f"  Top10 + Bottom10 panels: {vis_dir}/")
    print(f"  Per-slice CSV: {csv_path}")

    # 7. 也保存完整 3D nii (可选)
    # 不做反向变换，直接保存 crop 空间的预测 -> 简单但无法叠加原图
    # 这里更实用: 保存全尺寸 nii.gz 供 ITK-SNAP
    print(f"\n  Saving 3D nifti...")
    h_full, w_full, d_full = lbl_data.shape
    pred_full = np.zeros((z_dim, h_full, w_full), dtype=np.uint8)
    gt_full = np.zeros((z_dim, h_full, w_full), dtype=np.uint8)
    img_full = np.zeros((z_dim, h_full, w_full), dtype=np.uint8)

    for z, img_slc, gt_slc, pred_slc, d_p, d_t in records:
        h_a = min(PATCH_SIZE, h_full - y_start)
        w_a = min(PATCH_SIZE, w_full - x_start)
        pred_full[z, y_start:y_start+h_a, x_start:x_start+w_a] = pred_slc[:h_a, :w_a]
        gt_full[z, y_start:y_start+h_a, x_start:x_start+w_a] = gt_slc[:h_a, :w_a]
        img_full[z, y_start:y_start+h_a, x_start:x_start+w_a] = (img_slc[:h_a, :w_a] * 255).astype(np.uint8)

    # 反向旋转
    def rev_transform(vol):
        """逆变换: (fliplr∘rot90(k=1))^{-1} = rot90(k=3)∘fliplr"""
        out = [np.fliplr(vol[i]) for i in range(vol.shape[0])]
        out = [np.rot90(o, k=3) for o in out]
        return np.stack(out, axis=0)

    pred_rev = rev_transform(pred_full)
    gt_rev = rev_transform(gt_full)
    img_rev = rev_transform(img_full)

    pred_nii = nibabel.Nifti1Image(pred_rev.transpose(1,2,0).astype(np.float32), None, header)
    gt_nii   = nibabel.Nifti1Image(gt_rev.transpose(1,2,0).astype(np.float32), None, header)
    img_nii  = nibabel.Nifti1Image(img_rev.transpose(1,2,0).astype(np.uint8), None, header)

    nibabel.save(pred_nii, os.path.join(vis_dir, f'{patient_id}_pred.nii.gz'))
    nibabel.save(gt_nii,   os.path.join(vis_dir, f'{patient_id}_gt.nii.gz'))
    nibabel.save(img_nii,  os.path.join(vis_dir, f'{patient_id}_img.nii.gz'))

    print(f"  Saved: {vis_dir}/{patient_id}_*.nii.gz")
    print(f"\n  All done!")


if __name__ == '__main__':
    main()
