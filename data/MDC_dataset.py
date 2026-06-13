from io import BytesIO
import random
import logging
import warnings
import os
import numpy as np
import torch
import utils.transforms as T
import math
from PIL import Image, ImageFile

from data import data_utils
from data.base_dataset import BaseDataset
from bert.tokenization_bert import BertTokenizer
from data.poly_utils import string_to_polygons, downsample_polygons, polygons_to_string, points_to_token_string
import cv2
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class MedicalPolyDataset(BaseDataset):
    def __init__(
            self,
            split,
            data_root,
            pickle_path,
            bpe,
            src_dict,
            tgt_dict,
            max_src_length=80,
            num_bins=1000,
            max_image_size=512,
            n_splits=5,  # [新增] 交叉验证的总折数
            fold=0  # [新增] 当前运行的验证折索引 (0~4)
    ):
        super().__init__(split, None, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.num_bins = num_bins
        self.data_root = data_root
        self.img_size = max_image_size
        # 1. 瞬间将所有的 Prompt 和 Polygons 载入内存
        with open(pickle_path, 'rb') as f:
            self.annotations = pickle.load(f)

        # 2. 构建扁平化索引表：(patient_id, z_idx)
        self.index_list = []

        # 统计计数（可选，方便打印日志）
        patient_count = 0
        slice_count = 0

        for pid, info in self.annotations.items():
            # [核心逻辑]: 提取 PID 中的数字，例如 "pancreas_001" -> 1
            num_str = ''.join(filter(str.isdigit, pid))
            if not num_str:
                continue
            case_id = int(num_str)

            # 判断当前 case_id 是否属于验证集所在折
            is_val_fold = (case_id % n_splits) == fold

            # [拦截过滤]:
            # 如果当前是构建 'train' 数据集，但该病人属于 val 折，则跳过
            if split == 'train' and is_val_fold:
                continue
            # 如果当前是构建 'valid' (或 val) 数据集，但该病人不属于 val 折，则跳过
            if split != 'train' and not is_val_fold:
                continue

            patient_count += 1

            for i, slice_info in enumerate(info["slices_info"]):
                # 剔除空白层
                if len(slice_info["polygons"]) > 0:
                    self.index_list.append((pid, i))
                    slice_count += 1

        print(
            f"[{split.upper()} Set] Fold {fold}/{n_splits}: Loaded {patient_count} patients, {slice_count} valid slices.")

        self.positioning_transform = T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/pretrained_weights/RadBERT/')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        # patient_id 是病人名，item_idx 是在 slices_info 列表中的序号
        patient_id, item_idx = self.index_list[index]
        anno = self.annotations[patient_id]

        # 关键步骤：拿到该样本对应的具体切片信息
        slice_info = anno["slices_info"][item_idx]

        # 获取预处理时保存的真实 Z 轴物理索引
        z_idx = slice_info["relative_z"]

        # 1. 极速读取解耦的图像数据
        img_path = os.path.join(self.data_root, anno["img_path"])
        mask_path = os.path.join(self.data_root, anno["mask_path"])
        slice_mask = np.load(mask_path, mmap_mode='r')[z_idx]
        slice_img = np.load(img_path, mmap_mode='r')[z_idx]  # shape: (N, 256, 256, 3)
        # ---------------------------------------------------------
        # 核心修改：直接进行高效的 Tensor 转换和维度调整
        # ---------------------------------------------------------
        # 1. 转为 Tensor 并变换通道为 (C, H, W) -> (3, 256, 256)
        # copy() 是为了防止 mmap_mode 下的内存地址不连续报错
        patch_image = torch.from_numpy(slice_img.copy()).float().permute(2, 0, 1)
        patch_mask = torch.tensor([True])
        w, h = self.img_size, self.img_size
        # patch_image = self.positioning_transform(patch_image, target=None)

        polygons = slice_info["polygons"]
        task = slice_info["task"]

        if self.split == 'train':
            text_prompt = random.choice(slice_info["prompts"])
        else:
            text_prompt = slice_info["prompts"][0]

        src_caption = self.pre_caption(text_prompt, self.max_src_length)
        prompt = ' which region does the text " {} " describe?'.format(src_caption)

        # ========================================================
        # 4. 坐标处理与量子化
        # ========================================================
        polygons_scaled = []
        for polygon in polygons:
            polygon = np.array(polygon, dtype=np.float32)
            n_point = len(polygon) // 2
            scale = np.concatenate([np.array([w, h]) for _ in range(n_point)], 0)
            poly_scaled = (polygon / scale).reshape(n_point, 2)
            polygons_scaled.append(poly_scaled)

        # --- 步骤 1: 计算真实 Bbox ---
        if len(polygons) > 0:
            # 合并所有多边形点以计算边界
            all_points = np.concatenate([np.array(p).reshape(-1, 2) for p in polygons])
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            # 缩放到 [0, 1]
            region_points = torch.tensor([[x_min / self.img_size, y_min / self.img_size],
                                          [x_max / self.img_size, y_max / self.img_size]], dtype=torch.float32)
        else:
            region_points = torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=torch.float32)

        region = np.array([x_min, y_min, x_max, y_max])

        quant_box = region_points * (self.num_bins - 1)
        quant_box11 = [[math.floor(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box21 = [[math.ceil(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box12 = [[math.floor(p[0]), math.ceil(p[1])] for p in quant_box]
        quant_box22 = [[math.ceil(p[0]), math.ceil(p[1])] for p in quant_box]

        quant_poly = [poly * (self.num_bins - 1) for poly in polygons_scaled]
        quant_poly11 = [[[math.floor(p[0]), math.floor(p[1])] for p in poly] for poly in quant_poly]
        quant_poly21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in poly] for poly in quant_poly]
        quant_poly12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in poly] for poly in quant_poly]
        quant_poly22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in poly] for poly in quant_poly]

        # ========================================================
        # 1. 严格使用源码的字符串转换函数获取坐标串和 token_type
        # ========================================================
        region_coord11, _ = points_to_token_string(quant_box11, quant_poly11)
        region_coord21, _ = points_to_token_string(quant_box21, quant_poly21)
        region_coord12, _ = points_to_token_string(quant_box12, quant_poly12)
        region_coord22, token_type = points_to_token_string(quant_box22, quant_poly22)

        # ========================================================
        # 2. 严格使用源码计算 Delta 插值系数[cite: 4]
        # ========================================================
        delta_x1 = [0] + [p[0] - math.floor(p[0]) for p in quant_box]  # [0] for bos token
        for polygon in quant_poly:
            delta = [poly_point[0] - math.floor(poly_point[0]) for poly_point in polygon]
            delta_x1.extend(delta)
            delta_x1.extend([0])  # for separator token
        delta_x1 = delta_x1[:-1]  # there is no separator token in the end
        delta_x1 = torch.tensor(delta_x1, dtype=torch.float32)
        delta_x2 = 1 - delta_x1

        delta_y1 = [0] + [p[1] - math.floor(p[1]) for p in quant_box]  # [0] for bos token
        for polygon in quant_poly:
            delta = [poly_point[1] - math.floor(poly_point[1]) for poly_point in polygon]
            delta_y1.extend(delta)
            delta_y1.extend([0])  # for separator token
        delta_y1 = delta_y1[:-1]  # there is no separator token in the end
        delta_y1 = torch.tensor(delta_y1, dtype=torch.float32)
        delta_y2 = 1 - delta_y1

        token_type.append(2)  # 2 for eos token

        # ========================================================
        # 3. 严格使用源码的 encode_text 映射真实的 Token ID[cite: 4]
        # ========================================================
        tgt_item11 = self.encode_text(region_coord11, use_bpe=False)
        tgt_item12 = self.encode_text(region_coord12, use_bpe=False)
        tgt_item21 = self.encode_text(region_coord21, use_bpe=False)
        tgt_item22 = self.encode_text(region_coord22, use_bpe=False)

        target_item = region_points
        for poly in polygons_scaled:
            target_item = torch.cat([target_item, torch.tensor(poly, dtype=torch.float32), torch.tensor([[0.0, 0.0]])],
                                    dim=0)

        # ========================================================
        # 5. 严格按照源码构建输入端 prev_output (拼接 BOS)[cite: 4]
        # ========================================================
        prev_output_item11 = torch.cat([torch.tensor([self.bos_item]), tgt_item11])
        prev_output_item12 = torch.cat([torch.tensor([self.bos_item]), tgt_item12])
        prev_output_item21 = torch.cat([torch.tensor([self.bos_item]), tgt_item21])
        prev_output_item22 = torch.cat([torch.tensor([self.bos_item]), tgt_item22])

        # ========================================================
        # 6. 组装字典 (键名与原版严格保持一致)[cite: 4]
        # ========================================================
        example = {
            "id": f"{patient_id}_{z_idx}",
            "source": prompt,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,  # 👈 源码原汁原味的浮点目标[cite: 4]
            "prev_output_tokens_11": prev_output_item11,  # 👈 源码原汁原味的整数序列[cite: 4]
            "prev_output_tokens_12": prev_output_item12,
            "prev_output_tokens_21": prev_output_item21,
            "prev_output_tokens_22": prev_output_item22,
            "delta_x1": delta_x1,
            "delta_y1": delta_y1,
            "delta_x2": delta_x2,
            "delta_y2": delta_y2,
            "w_resize_ratio": torch.tensor(1.0),
            "h_resize_ratio": torch.tensor(1.0),
            "region_coord": torch.tensor(region),
            "token_type": torch.tensor(token_type),
            "w": torch.tensor(w),
            "h": torch.tensor(h),
            "label": torch.tensor(slice_mask, dtype=torch.uint8),
            "n_poly": len(polygons),
            "text": src_caption,
            "task": task
        }
        return example

    def collate(self, samples, pad_idx, eos_idx):
        if len(samples) == 0:
            return {}

        def merge(key, padding_item):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                padding_item,
                eos_idx=eos_idx,
            )

        id = np.array([s["id"] for s in samples])
        captions = [s["source"] for s in samples]
        tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt")
        src_tokens = tokenized["input_ids"]
        att_masks = tokenized["attention_mask"]
        src_lengths = torch.LongTensor(att_masks.ne(0).long().sum())

        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
        h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)

        delta_x1 = merge("delta_x1", 0)
        delta_y1 = merge("delta_y1", 0)
        delta_x2 = merge("delta_x2", 1)
        delta_y2 = merge("delta_y2", 1)

        # 【核心修正】: 取消注释，完美对齐原版的 collate 逻辑
        region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

        target = merge("target", pad_idx)
        tgt_lengths = torch.LongTensor([s["target"].shape[0] for s in samples])
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens_11 = merge("prev_output_tokens_11", pad_idx)
        prev_output_tokens_12 = merge("prev_output_tokens_12", pad_idx)
        prev_output_tokens_21 = merge("prev_output_tokens_21", pad_idx)
        prev_output_tokens_22 = merge("prev_output_tokens_22", pad_idx)

        # 【核心修正】: 取消注释，完美对齐原版的 collate 逻辑
        token_type = merge("token_type", -1)
        w = torch.stack([s["w"] for s in samples], dim=0)
        h = torch.stack([s["h"] for s in samples], dim=0)
        n_poly = [s['n_poly'] for s in samples]

        labels = np.stack([sample['label'] for sample in samples], 0)
        text = [s["text"] for s in samples]
        ttasks = [s["task"] for s in samples]

        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "att_masks": att_masks,
                "patch_images": patch_images,
                "patch_masks": patch_masks,
                "prev_output_tokens_11": prev_output_tokens_11,
                "prev_output_tokens_12": prev_output_tokens_12,
                "prev_output_tokens_21": prev_output_tokens_21,
                "prev_output_tokens_22": prev_output_tokens_22,
                "delta_x1": delta_x1,
                "delta_y1": delta_y1,
                "delta_x2": delta_x2,
                "delta_y2": delta_y2
            },
            "target": target,
            "w_resize_ratios": w_resize_ratios,
            "h_resize_ratios": h_resize_ratios,
            "region_coords": region_coords,
            "label": labels,
            "token_type": token_type,
            "w": w,
            "h": h,
            "n_poly": n_poly,
            "text": text,
            "task": ttasks,
        }

        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)


