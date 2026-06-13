#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from fairseq import (
    # checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
# from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from utils import checkpoint_utils
from trainer import Trainer
import cv2


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    cfg.checkpoint.no_epoch_checkpoints = True
    cfg.checkpoint.save_interval = 999999
    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    # data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=True,
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    if max_epoch > 0 and max_epoch != math.inf:
        total_num_updates = sum(
            math.ceil(len(epoch_itr) / cfg.optimization.update_freq[i])
            if i < len(cfg.optimization.update_freq) else
            math.ceil(len(epoch_itr) / cfg.optimization.update_freq[-1])
            for i in range(max_epoch)
        )
        trainer.lr_reinit(total_num_updates, trainer.get_num_updates())
    lr = trainer.get_lr()

    # =================================================================
    # 🚨 【DEBUG】: 强行拦截！只跑 Validation 测试代码，跑完直接退出
    # =================================================================
    # logger.info("⚠️ [DEBUG MODE] 正在跳过训练，直接执行 Validation 测试 pipeline...")
    # valid_subsets = cfg.dataset.valid_subset.split(",")
    #
    # # 手动调用我们刚才修改过的 validate 函数
    # try:
    #     validate(cfg, trainer, task, epoch_itr, valid_subsets)
    #     logger.info("✅ [DEBUG MODE] Validation 测试完美跑通！图片应该已经保存了。")
    # except Exception as e:
    #     logger.error(f"❌ [DEBUG MODE] Validation 阶段崩溃，请看报错：\n{e}")
    #     raise e
    #
    # import sys
    # sys.exit(0)  # 测试完毕，直接终止程序，不进入下方漫长的 train 循环
    # =================================================================

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, 0)
        #lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=True,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=True,
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):

        # ==========================================================
        # 🚨 [DEBUG] 拦截打印！(直接粘贴在这里)
        # ==========================================================
        # if i == 0:  # 仅在每个 Epoch 的第一个 Batch 打印
        #     try:
        #         import logging
        #         debug_logger = logging.getLogger(__name__)
        #
        #         # 兼容不同版本的 fairseq 返回单数还是复数
        #         current_sample = samples[0] if isinstance(samples, list) else samples
        #
        #         if 'target' in current_sample:
        #             target_tensor = current_sample['target'][0]
        #             tokens = target_tensor.cpu().numpy().tolist()
        #
        #             debug_logger.info("\n" + "🔥" * 30)
        #             debug_logger.info("🎯 [DEBUG] 检查 Ground Truth (GT) Token 序列:")
        #             debug_logger.info(f"👉 序列前 100 个 Token: \n{tokens[:100]}")
        #
        #             coord_tokens = [t for t in tokens if t >= 4]
        #             if len(coord_tokens) > 0:
        #                 min_val, max_val = min(coord_tokens), max(coord_tokens)
        #                 debug_logger.info(f"📊 坐标 Token 统计 -> 最小值: {min_val}, 最大值: {max_val}")
        #                 if max_val > 1050:
        #                     debug_logger.error("🚨 致命警告：发现极其巨大的 Token ID！量化出错了！")
        #             else:
        #                 debug_logger.warning("🚨 警告：没有找到有效的坐标 Token (>=4)！全是控制符？")
        #             debug_logger.info("🔥" * 30 + "\n")
        #     except Exception as e:
        #         pass
        # ==========================================================

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()

        # if task.cfg._name == 'refcoco_pretrain':
        #     valid_losses, should_stop = validate_and_save(
        #         cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        #     )
        # else:
        #     # skip validation during training in fine-tuning stage
        #     valid_losses = 0
        #     should_stop = False
        # if should_stop:
        #     break

        # [修改点 1]：移除原版的硬编码限制，强制在每个 Epoch 结束时执行验证和保存
        if end_of_epoch:
            valid_losses, should_stop = validate_and_save(
                cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
            )
        else:
            valid_losses = [None]
            should_stop = False

        if should_stop:
            break

    # checkpoint_utils.save_checkpoint(
    #     cfg.checkpoint, trainer, epoch_itr, 0
    # )
    if task.cfg._name == 'refcoco':
        cmd = f'cp {cfg.checkpoint.save_dir}/checkpoint_last.pt {cfg.checkpoint.save_dir}/checkpoint_epoch_{epoch_itr.epoch}.pt'
        print(cmd)
        os.system(cmd)
    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # 1. 判断是否该停止训练
    should_stop = num_updates >= max_update

    # 2. 判断是否需要验证和保存
    # (如果到了 epoch 结束，我们就执行验证和保存)
    do_save = end_of_epoch or should_stop
    do_validate = (end_of_epoch or should_stop) and not cfg.dataset.disable_validation

    # 3. 执行验证
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # 4. 执行保存 (🚨 核心：无条件调用底层保存函数)
    if do_save or should_stop:
        # 把评估指标 (valid_losses[0]) 直接扔给底层
        # Fairseq 会自动根据你的 .sh 参数去判断它是不是 best！
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop

"""
def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop
"""

def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )
        # [修改点 3]：初始化 TP, FP, FN 累加器，借鉴评价脚本逻辑
        pancreas_tp, pancreas_fp, pancreas_fn = 0, 0, 0
        tumor_tp, tumor_fp, tumor_fn = 0, 0, 0

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break

                # 1. 执行验证前向传播
                trainer.valid_step(sample)

                # 2. 从任务中获取预测的多边形坐标和真实的掩码
                # （这里假设你在 task.valid_step 中实现了将坐标返回或保存在某个属性中）
                # 理论上需要通过模型推理得到 hyps (预测) 和 refs (真实)，并还原为 mask
                if hasattr(task, 'get_predictions_and_masks'):
                    pred_masks, gt_masks = task.get_predictions_and_masks(trainer.get_model(), sample)
                sample_ids = sample.get('id', [f"batch_{i}_idx_{j}" for j in range(len(pred_masks))])

                for j, (p_mask, g_mask) in enumerate(zip(pred_masks, gt_masks)):
                    # ====================================================
                    # A. 评价指标计算 (原版逻辑保持不变)
                    # ====================================================
                    panc_ref = (g_mask == 1) | (g_mask == 2)
                    panc_pred = (p_mask == 1)
                    pancreas_tp += np.sum(panc_ref & panc_pred)
                    pancreas_fp += np.sum((~panc_ref) & panc_pred)
                    pancreas_fn += np.sum(panc_ref & (~panc_pred))

                    tumor_ref = (g_mask == 2)
                    tumor_pred = (p_mask == 2)
                    tumor_tp += np.sum(tumor_ref & tumor_pred)
                    tumor_fp += np.sum((~tumor_ref) & tumor_pred)
                    tumor_fn += np.sum(tumor_ref & (~tumor_pred))

                    # ====================================================
                    # B. 【官方源码版】半透明叠加 + 锐利边缘可视化
                    # ====================================================
                    if i % 60 == 0 and j % 50 == 0:
                        current_epoch = epoch_itr.epoch

                        slice_id = sample_ids[j]

                        # 1. 提取原图
                        img_tensor = sample['net_input']['patch_images'][j]
                        orig_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
                        orig_img = orig_img - orig_img.min()
                        max_val = orig_img.max()
                        if max_val > 0:
                            orig_img = (orig_img / max_val * 255).astype(np.uint8)
                        else:
                            orig_img = orig_img.astype(np.uint8)
                        ttask = sample['task'][j]
                        prompt = sample['text'][j]
                        # 2. 引入官方 overlay_davis 函数 (略作颜色适配)
                        def overlay_davis(image, mask, colors, alpha=0.4):
                            from scipy.ndimage.morphology import binary_dilation
                            colors = np.reshape(colors, (-1, 3))
                            im_overlay = image.copy()
                            object_ids = np.unique(mask)

                            for object_id in object_ids[1:]:  # 跳过背景 0
                                # 半透明前景
                                foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(
                                    colors[object_id])
                                binary_mask = mask == object_id

                                # 填充半透明颜色
                                im_overlay[binary_mask] = foreground[binary_mask]

                                # 形态学计算轮廓，并描黑边增强对比度
                                countours = binary_dilation(binary_mask) ^ binary_mask
                                im_overlay[countours, :] = 0  # 黑色轮廓线

                            return im_overlay.astype(image.dtype)

                        # 3. 颜色映射表 (OpenCV BGR 格式)
                        # 0: 背景(忽略), 1: 胰腺(红色), 2: 肿瘤(绿色)
                        davis_colors = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]

                        # 4. 渲染
                        gt_overlay = overlay_davis(orig_img, g_mask.astype(np.uint8), davis_colors, alpha=0.35)
                        pred_overlay = overlay_davis(orig_img, p_mask.astype(np.uint8), davis_colors, alpha=0.35)

                        # 5. 左右拼接与文字标注
                        vis_image = np.concatenate([gt_overlay, pred_overlay], axis=1)

                        # 背景框
                        cv2.rectangle(vis_image, (10, 5), (150, 30), (0, 0, 0), -1)
                        cv2.rectangle(vis_image, (512 + 10, 5), (512 + 130, 30), (0, 0, 0), -1)
                        # cv2.rectangle(vis_image, (10, 225), (120, 250), (0, 0, 0), -1)

                        cv2.putText(vis_image, "Ground Truth", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                    1, cv2.LINE_AA)
                        cv2.putText(vis_image, "Prediction", (512 + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        text_size, _ = cv2.getTextSize(f"{prompt}", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        box_w = text_size[0] + 10
                        box_h = text_size[1] + 15
                        cv2.rectangle(vis_image, (10, 500), (10 + box_w, 500+box_h), (0, 0, 0), -1)
                        cv2.putText(vis_image, f"P{prompt}", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                                    1, cv2.LINE_AA)

                        save_path = f"{cfg.checkpoint.save_dir}/epoch{current_epoch}_{slice_id}_{ttask}.jpg"
                        # 1. 写入图片
                        cv2.imwrite(save_path, vis_image)

            # [修改点 4]：计算最终的 Dice 和 IoU 并在日志中输出

        def calc_metrics(tp, fp, fn):
            if tp + fp + fn == 0: return 0.0, 0.0
            iou = tp / (tp + fp + fn)
            dice = 2 * tp / (2 * tp + fp + fn)
            return iou, dice

        p_iou, p_dice = calc_metrics(pancreas_tp, pancreas_fp, pancreas_fn)
        t_iou, t_dice = calc_metrics(tumor_tp, tumor_fp, tumor_fn)

        # 记录到 Fairseq 的 metrics 系统中
        metrics.log_scalar("val_panc_iou", p_iou, priority=10, round=4)
        metrics.log_scalar("val_panc_dice", p_dice, priority=11, round=4)
        metrics.log_scalar("val_tumor_iou", t_iou, priority=12, round=4)
        metrics.log_scalar("val_tumor_dice", t_dice, priority=13, round=4)

        # log validation stats
        if hasattr(task, 'get_valid_stats'):
            stats = task.get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        else:
            stats = agg.get_smoothed_values()

        # 将我们计算的指标强行塞入最终显示的 stats 字典中
        stats["panc_dice"] = round(p_dice, 4)
        stats["tumor_dice"] = round(t_dice, 4)
        stats = get_valid_stats(cfg, trainer, stats)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    parser.add_argument("--det_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
