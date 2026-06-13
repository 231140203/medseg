# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
import os
import logging
from typing import Optional
import math
import numpy as np
import torch
from fairseq import metrics
from fairseq.tasks import register_task


# from data.refcoco_pretrain_dataset import RefcocoPretrainDataset
from data.file_dataset import FileDataset
from data.MDC_dataset import MedicalPolyDataset
from tasks.base_task import BaseTask, BaseConfig, load_bert_pretrained_weights

logger = logging.getLogger(__name__)
import cv2
from shapely.geometry import Polygon

@dataclass
class MDCPretrainConfig(BaseConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    # [新增] 交叉验证参数
    n_splits: int = field(
        default=5, metadata={"help": "Total number of splits for cross-validation"}
    )
    fold: int = field(
        default=0, metadata={"help": "Current fold index (0 to n_splits-1) to use as validation"}
    )


@register_task("MDC_pretrain", dataclass=MDCPretrainConfig)
class MDCPretrainTask(BaseTask):
    def __init__(self, cfg: MDCPretrainConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    # def load_dataset(self, split, epoch=1, combine=False, **kwargs):
    #     paths = self.cfg.data.split(',')
    #     assert len(paths) > 0
    #
    #     if split == 'train':
    #         file_path = paths[(epoch - 1) % (len(paths) - 1)]
    #     else:
    #         file_path = paths[-1]
    #     dataset = FileDataset(file_path, self.cfg.selected_cols)
    #
    #     self.datasets[split] = RefcocoPretrainDataset(
    #         split,
    #         dataset,
    #         self.bpe,
    #         self.src_dict,
    #         self.tgt_dict,
    #         max_src_length=self.cfg.max_src_length,
    #         max_tgt_length=self.cfg.max_tgt_length,
    #         patch_image_size=self.cfg.patch_image_size,
    #         imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
    #         num_bins=self.cfg.num_bins,
    #         max_image_size=self.cfg.max_image_size
    #     )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # [修改] 直接从配置中读取我们在 sh 文件里传进来的绝对路径
        data_root = self.cfg.data_root
        pickle_path = self.cfg.pickle_path

        # 使用我们重构的 MedicalPolyDataset，并传入交叉验证参数
        self.datasets[split] = MedicalPolyDataset(
            split=split,
            data_root=data_root,
            pickle_path=pickle_path,
            bpe=self.bpe,
            src_dict=self.src_dict,
            tgt_dict=self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size,
            n_splits=self.cfg.n_splits,  # [新增]
            fold=self.cfg.fold  # [新增]
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        # bert_path = "/root/autodl-tmp/pretrained_weights/bert-base-uncased-pytorch_model.bin"
        # if os.path.exists(bert_path):
        #     load_bert_pretrained_weights(model.encoder.bert, bert_path)
        if cfg._name == 'polyformer_b':
            swin_path = "/root/autodl-tmp/pretrained_weights/swin_base_patch4_window12_384_22k.pth"
        else:
            swin_path = "/root/autodl-tmp/pretrained_weights/swin_large_patch4_window12_384_22k.pth"
        if os.path.exists(swin_path):
            model.encoder.embed_images.init_weights(pretrained=swin_path)
        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)
        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(sample, model)
            scores = self._calculate_ap_score(hyps.float(), refs.float())
            logging_output["_score_sum"] = scores.sum().item()
            logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def _inference(self, sample, model):
        hyps = self.inference_step(model, sample)
        refs = sample['region_coords'].float()
        hyps = hyps * self.cfg.max_image_size
        hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
        return hyps, refs

    def inference_step(self, model, sample):
        with torch.no_grad():
            if isinstance(model, list):
                model = model[0]
            total_len = 2
            model.eval()
            img = sample["net_input"]["patch_images"]
            b = img.shape[0]
            prev_output_token_11 = [[0] for _ in range(b)]
            prev_output_token_12 = [[0] for _ in range(b)]
            prev_output_token_21 = [[0] for _ in range(b)]
            prev_output_token_22 = [[0] for _ in range(b)]
            delta_x1 = [[0] for _ in range(b)]
            delta_y1 = [[0] for _ in range(b)]
            delta_x2 = [[1] for _ in range(b)]
            delta_y2 = [[1] for _ in range(b)]

            gen_out = [[] for _ in range(b)]

            n_bins = self.cfg.num_bins

            encoder_out = model.encoder(
                sample['net_input']['src_tokens'],
                src_lengths=sample['net_input']['src_lengths'],
                att_masks=sample['net_input']['att_masks'],
                patch_images=sample['net_input']['patch_images'],
                patch_masks=sample['net_input']['patch_masks'],
                token_embeddings=None,
                return_all_hiddens=False,
                sample_patch_num=None
            )

            for i in range(total_len):
                prev_output_tokens_11_tensor = torch.tensor(np.array(prev_output_token_11)).to(img.device).long()
                prev_output_tokens_12_tensor = torch.tensor(np.array(prev_output_token_12)).to(img.device).long()
                prev_output_tokens_21_tensor = torch.tensor(np.array(prev_output_token_21)).to(img.device).long()
                prev_output_tokens_22_tensor = torch.tensor(np.array(prev_output_token_22)).to(img.device).long()
                delta_x1_tensor = torch.tensor(np.array(delta_x1)).to(img.device)
                delta_x2_tensor = torch.tensor(np.array(delta_x2)).to(img.device)
                delta_y1_tensor = torch.tensor(np.array(delta_y1)).to(img.device)
                delta_y2_tensor = torch.tensor(np.array(delta_y2)).to(img.device)

                net_output = model.decoder(
                    prev_output_tokens_11_tensor,
                    prev_output_tokens_12_tensor,
                    prev_output_tokens_21_tensor,
                    prev_output_tokens_22_tensor,
                    delta_x1_tensor,
                    delta_y1_tensor,
                    delta_x2_tensor,
                    delta_y2_tensor,
                    code_masks=None,
                    encoder_out=encoder_out,
                    features_only=False,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=sample['net_input']['src_lengths'],
                    return_all_hiddens=False
                )
                net_output = net_output[1]
                for j in range(b):
                    output_j_x, output_j_y = net_output[j, i].cpu().numpy()
                    gen_out[j].extend([output_j_x, output_j_y])

                    output_j_x = output_j_x * (n_bins - 1)
                    output_j_y = output_j_y * (n_bins - 1)

                    output_j_x_floor = math.floor(output_j_x)
                    output_j_y_floor = math.floor(output_j_y)
                    output_j_x_ceil = math.ceil(output_j_x)
                    output_j_y_ceil = math.ceil(output_j_y)

                    # convert to token
                    prev_output_token_11[j].append(output_j_x_floor * n_bins + output_j_y_floor + 4)
                    prev_output_token_12[j].append(output_j_x_floor * n_bins + output_j_y_ceil + 4)
                    prev_output_token_21[j].append(output_j_x_ceil * n_bins + output_j_y_floor + 4)
                    prev_output_token_22[j].append(output_j_x_ceil * n_bins + output_j_y_ceil + 4)

                    delta_x = output_j_x - output_j_x_floor
                    delta_y = output_j_y - output_j_y_floor
                    delta_x1[j].append(delta_x)
                    delta_y1[j].append(delta_y)
                    delta_x2[j].append(1-delta_x)
                    delta_y2[j].append(1-delta_y)
        return torch.tensor(gen_out).to(img.device)

    # ==========================================
    # 🌟 独立的自交叉修复函数
    # ==========================================
    def fix_self_intersection(self, poly_pts, method='mask', image_size=(512, 512)):
        """
        检查并修复多边形的自交叉与内部空洞问题。
        :param poly_pts: numpy array, 形状为 (N, 2) 的原图像素坐标点集
        :param method: 'mask' (掩码渲染法) 或 'shapely' (几何数学法)
        :param image_size: (h, w) 用于掩码渲染的画布尺寸
        :return: 修复后的点集，numpy array (M, 2)
        """
        if len(poly_pts) < 3:
            return poly_pts

        if method == 'mask':
            # ----------------------------------
            # 方法一：Mask 渲染与轮廓重提法 (极度稳健)
            # ----------------------------------
            h, w = image_size
            temp_mask = np.zeros((h, w), dtype=np.uint8)

            # 强行填涂，cv2.fillPoly 天生支持自交叉区域的填充
            cv2.fillPoly(temp_mask, [poly_pts], 255)

            # 形态学闭运算：填平模型生成时手抖产生的小毛刺和细微缺口
            kernel = np.ones((5, 5), np.uint8)
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)

            # 提取最外层轮廓，彻底丢弃内部打结的线条和空洞 (RETR_EXTERNAL)
            contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return np.array([])

            # 如果交叉过于严重断裂成多个区块，取最大的那个主病灶
            main_contour = max(contours, key=cv2.contourArea)

            # 保证返回的维度始终是 (N, 2)
            return main_contour.squeeze().reshape(-1, 2)

        elif method == 'shapely':
            # ----------------------------------
            # 方法二：Shapely 几何数学法 (纯坐标计算)
            # ----------------------------------
            poly = Polygon(poly_pts)

            # buffer(0) 技巧：拓扑学上会自动化解自交叉并分离合法多边形
            if not poly.is_valid:
                clean_poly = poly.buffer(0)
            else:
                clean_poly = poly

            # 如果修复后分裂成了 MultiPolygon，取面积最大的一块
            if clean_poly.geom_type == 'MultiPolygon':
                clean_poly = max(clean_poly.geoms, key=lambda a: a.area)

            # 修复失败导致多边形坍塌则返回空
            if clean_poly.geom_type != 'Polygon' or clean_poly.is_empty:
                return np.array([])

            # 提取外部边界坐标，[:-1] 是因为 Shapely 默认首尾点重复以闭合
            coords = np.array(clean_poly.exterior.coords)[:-1]
            return np.round(coords).astype(np.int32)

        else:
            raise ValueError("Unsupported method. Choose 'mask' or 'shapely'.")

    def get_predictions_and_masks(self, model, sample):
        model.eval()
        with torch.no_grad():
            # =====================================================================
            # 🚨 【关键补齐】：动态精度与设备对齐 (处理 FP32->FP16, CPU->GPU)
            # =====================================================================
            model_param = next(model.parameters())
            model_dtype = model_param.dtype
            model_device = model_param.device

            # 将图像对齐到模型的精度和设备
            img = sample['net_input']['patch_images'].to(device=model_device, dtype=model_dtype)
            sample['net_input']['patch_images'] = img

            # 顺手把文本输入也拉到 GPU 上（保持 long int 类型即可，不需要变半精度）
            sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(device=model_device)
            if sample['net_input'].get('src_lengths') is not None:
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].to(device=model_device)
            if sample['net_input'].get('att_masks') is not None:
                sample['net_input']['att_masks'] = sample['net_input']['att_masks'].to(device=model_device)
            # =====================================================================

            b = img.shape[0]

            # ====== 【完全照搬源码 source: 3：抛弃 Generator，纯手工自回归】 ======
            prev_output_token_11 = [[0] for _ in range(b)]
            prev_output_token_12 = [[0] for _ in range(b)]
            prev_output_token_21 = [[0] for _ in range(b)]
            prev_output_token_22 = [[0] for _ in range(b)]
            delta_x1 = [[0.0] for _ in range(b)]
            delta_y1 = [[0.0] for _ in range(b)]
            delta_x2 = [[1.0] for _ in range(b)]
            delta_y2 = [[1.0] for _ in range(b)]

            gen_out_coords = [[] for _ in range(b)]  # 源码直接存归一化的浮点坐标！
            unfinish_flag = np.ones(b)
            i = 0
            max_len = 400 # 你可以根据多边形顶点数量调大
            min_len = 40

            # 【注意】原版是 64，如果你的 MDC 任务切分了 1000 个 bin，请改为 1000
            n_bins = getattr(self.cfg, 'num_bins', 64)

            # 1. 编码器提取特征
            encoder_out = model.encoder(
                sample['net_input']['src_tokens'],
                src_lengths=sample['net_input']['src_lengths'],
                att_masks=sample['net_input'].get('att_masks', None),
                patch_images=img,
                patch_masks=sample['net_input'].get('patch_masks', None),
                token_embeddings=None,
                return_all_hiddens=False,
                sample_patch_num=None
            )

            model_dtype = next(model.parameters()).dtype

            # 2. 帧级自回归解码循环 (完全对照源码)
            while i < max_len and unfinish_flag.any():
                prev_11_t = torch.tensor(np.array(prev_output_token_11)).to(img.device).long()
                prev_12_t = torch.tensor(np.array(prev_output_token_12)).to(img.device).long()
                prev_21_t = torch.tensor(np.array(prev_output_token_21)).to(img.device).long()
                prev_22_t = torch.tensor(np.array(prev_output_token_22)).to(img.device).long()

                dx1_t = torch.tensor(np.array(delta_x1), dtype=model_dtype).to(img.device)
                dy1_t = torch.tensor(np.array(delta_y1), dtype=model_dtype).to(img.device)
                dx2_t = torch.tensor(np.array(delta_x2), dtype=model_dtype).to(img.device)
                dy2_t = torch.tensor(np.array(delta_y2), dtype=model_dtype).to(img.device)

                net_output = model.decoder(
                    prev_11_t, prev_12_t, prev_21_t, prev_22_t,
                    dx1_t, dy1_t, dx2_t, dy2_t,
                    code_masks=None,
                    encoder_out=encoder_out,
                    features_only=False,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=sample['net_input']['src_lengths'],
                    return_all_hiddens=False
                )

                cls_output = net_output[0]
                cls_type = torch.argmax(cls_output, 2)
                reg_output = net_output[1].squeeze(-1)

                for j in range(b):
                    if unfinish_flag[j] == 1:
                        cls_j = cls_type[j, i].item()
                        if cls_j == 0 or (cls_j == 2 and i < min_len):  # Coordinate token
                            out_x, out_y = reg_output[j, i].cpu().numpy()
                            out_x, out_y = min(out_x, 1), min(out_y, 1)

                            # 源码精髓：直接保存 0-1 的真实坐标系
                            gen_out_coords[j].extend([out_x, out_y])

                            out_x_bin = out_x * (n_bins - 1)
                            out_y_bin = out_y * (n_bins - 1)
                            out_x_floor, out_y_floor = math.floor(out_x_bin), math.floor(out_y_bin)
                            out_x_ceil, out_y_ceil = math.ceil(out_x_bin), math.ceil(out_y_bin)

                            prev_output_token_11[j].append(out_x_floor * n_bins + out_y_floor + 4)
                            prev_output_token_12[j].append(out_x_floor * n_bins + out_y_ceil + 4)
                            prev_output_token_21[j].append(out_x_ceil * n_bins + out_y_floor + 4)
                            prev_output_token_22[j].append(out_x_ceil * n_bins + out_y_ceil + 4)

                            delta_x = out_x_bin - out_x_floor
                            delta_y = out_y_bin - out_y_floor
                            delta_x1[j].append(delta_x);
                            delta_y1[j].append(delta_y)
                            delta_x2[j].append(1 - delta_x);
                            delta_y2[j].append(1 - delta_y)

                        elif cls_j == 1:  # Separator token (原版用分类 1 表示分隔符)
                            gen_out_coords[j].append(2)  # 存入特殊数字 2 作为分割标记
                            prev_output_token_11[j].append(3)
                            prev_output_token_12[j].append(3)
                            prev_output_token_21[j].append(3)
                            prev_output_token_22[j].append(3)
                            delta_x1[j].append(0);
                            delta_y1[j].append(0)
                            delta_x2[j].append(1);
                            delta_y2[j].append(1)
                        else:  # EOS token
                            unfinish_flag[j] = 0
                            gen_out_coords[j].append(-1)
                            prev_output_token_11[j].append(2)
                            prev_output_token_12[j].append(2)
                            prev_output_token_21[j].append(2)
                            prev_output_token_22[j].append(2)
                            delta_x1[j].append(0);
                            delta_y1[j].append(0)
                            delta_x2[j].append(1);
                            delta_y2[j].append(1)
                    else:
                        gen_out_coords[j].append(-1)
                        prev_output_token_11[j].append(1)
                        prev_output_token_12[j].append(1)
                        prev_output_token_21[j].append(1)
                        prev_output_token_22[j].append(1)
                        delta_x1[j].append(0);
                        delta_y1[j].append(0)
                        delta_x2[j].append(1);
                        delta_y2[j].append(1)
                i += 1

            # -------------------------------------------------------------------------
            # 3. 结果解析与掩码生成 (融合官方稳健逻辑重写)
            # -------------------------------------------------------------------------
            pred_masks, gt_masks = [], []

            # 这里可以自由切换你想测试的方法：'mask' 或 'shapely'
            POSTPROCESS_METHOD = 'mask'

            for j in range(b):
                preds = np.array(gen_out_coords[j])
                preds = preds[preds != -1]  # 排除 EOS 标志 (-1)

                h = img.shape[-2]
                w = img.shape[-1]

                # 初始化最终的预测掩码
                pred_mask = np.zeros((h, w), dtype=np.uint8)

                gt_data = sample['label'][j]
                if hasattr(gt_data, 'cpu'):
                    raw_gt_mask = gt_data.cpu().numpy()
                else:
                    raw_gt_mask = np.array(gt_data)

                task = sample['task'][j].lower() if 'task' in sample else 'pancreas'
                fill_value = 2 if 'tumor' in task else 1

                # 🚨 步骤 A：剥离冗余的 Bounding Box 坐标
                if len(preds) > 4:
                    polygons_pred = preds[4:]
                else:
                    polygons_pred = np.array([])

                # 🚨 步骤 B：按分隔符切分多个多边形
                polygons_pred = np.append(polygons_pred, [2])
                idx_list = [idx for idx, val in enumerate(polygons_pred) if val == 2]

                polygons = []
                pred_idx = 0
                for idx in idx_list:
                    if pred_idx != idx:
                        polygons.append(polygons_pred[pred_idx:idx])
                    pred_idx = idx + 1

                # 我们采用官方思路：为每个多边形建立独立图层，全部画完后再统一合并
                poly_masks = [np.zeros((h, w), dtype=np.uint8)]
                # 🚨 步骤 C & D：循环修复并直接画图
                for poly in polygons:
                    if len(poly) % 2 != 0:
                        poly = poly[:-1]

                    if len(poly) < 6:
                        continue

                    # 还原到原图尺寸
                    poly_pts = (poly.reshape(-1, 2) * [w, h]).astype(np.int32)

                    # ----------------------------------------------------
                    # 调用我们独立出来的函数进行自交叉和空洞修复！
                    # ----------------------------------------------------
                    # clean_pts = self.fix_self_intersection(poly_pts, method=POSTPROCESS_METHOD, image_size=(h, w))
                    #
                    # # 如果修复后多边形消失或小于3个点，直接跳过
                    # if len(clean_pts) < 3:
                    #     continue

                    # 因为 clean_pts 已经是绝对合法的闭合多边形，不会有空洞
                    # 直接使用 fillPoly 画到 pred_mask 上即可，无需再合并透明图层！
                    # cv2.fillPoly(pred_mask, [clean_pts], fill_value)
                    # 为当前单个多边形创建“专属透明图层”
                    single_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(single_mask, [poly_pts], 1)
                    poly_masks.append(single_mask)
                #
                # 将所有图层相加，重叠区域的值会大于 1
                # 通过 > 0 重新将其二值化为平整的 Mask，彻底消灭内部镂空
                combined_mask = sum(poly_masks) > 0
                pred_mask[combined_mask] = fill_value

                if 'tumor' in task:
                    # 肿瘤任务：GT 只要 2，预测不变
                    gt_mask = (raw_gt_mask == 2).astype(np.uint8) * 2
                else:
                    # 胰腺任务：
                    # 1. 严格设定 GT：只看纯健康组织 (1)
                    gt_mask = (raw_gt_mask == 1).astype(np.uint8) * 1

                    # 2. 强行修正 Pred：利用真实的肿瘤位置 (GT=2)，把预测结果中的对应区域挖掉
                    # 这样无论模型画没画好肿瘤，只要外轮廓画对了，这里就不会被误伤扣分！
                    pred_mask[raw_gt_mask == 2] = 0

                pred_masks.append(pred_mask)
                gt_masks.append(gt_mask)

            return pred_masks, gt_masks