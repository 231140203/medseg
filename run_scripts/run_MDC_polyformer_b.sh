#!/usr/bin/env bash

# ==========================================
# [修复点 1] 修复 libgomp 报错，强制限定线程数
export OMP_NUM_THREADS=1
export MASTER_PORT=6068
# [修改点 1] 仅使用第 0 号 GPU
export CUDA_VISIBLE_DEVICES=0
N_GPUS=1

num_bins=64
patch_image_size=512
n_splits=5
fold=0
det_weight=0.01
cls_weight=0.005
# [新增] 自动生成时间戳 (格式: 年月日_时分，例如 20240506_1430)
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# [进阶方案] 你也可以定义一个实验名，方便自己记忆
EXP_NAME="Stage1_pretrain512"
# ==========================================
# ==========================================
# [核心修改] 在这里填写你服务器上的真实绝对路径！
DATA_ROOT="/root/autodl-tmp/Data/Task07_Pancreas"
PICKLE_PATH=${DATA_ROOT}/MDC512_new_annotations.p
restore_file="/root/autodl-tmp/pretrained_weights/polyformer_b_pretrain.pt"
restore_file='/root/autodl-tmp/pretrained_weights/polyformer_radbert_pretrain.pt'
# ==========================================
save_dir="/root/autodl-tmp/MRIPolySeg"
log_dir=${save_dir}/Results/MDC/${EXP_NAME}/fold_${fold}/${TIMESTAMP}
ckp_dir=${save_dir}/Results/MDC/${EXP_NAME}/fold_${fold}/${TIMESTAMP}
mkdir -p $save_dir $log_dir $ckp_dir

bpe_dir=../utils/BPE
user_dir=../polyformer_module

task=MDC_pretrain
arch=polyformer_b
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=1e-4
max_epoch=20
warmup_ratio=0.06

# [修改点 2] 单卡显存有限，适当调小 batch_size，但利用 update_freq 累加梯度
batch_size=8
update_freq=6  # 等效 Batch Size = 4 * 4 * 1 = 16
max_src_length=80
max_tgt_length=400
echo "=========================================================="
echo "🚀 Starting Single-GPU Training for FOLD ${fold} / ${n_splits}"
echo "Data Root: ${DATA_ROOT}"
echo "Pickle Path: ${PICKLE_PATH}"
echo "Output Directory: ${save_dir}"
echo "=========================================================="

log_file=${log_dir}/${TIMESTAMP}_train.log

torchrun --nproc_per_node=${N_GPUS} --master_port=${MASTER_PORT} ../train.py \
    ./ \
    --data-root ${DATA_ROOT} \
    --pickle-path ${PICKLE_PATH} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --save-dir=${ckp_dir} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --warmup-ratio=${warmup_ratio} \
    --no-epoch-checkpoints \
    --keep-best-checkpoints=1 \
    --best-checkpoint-metric=tumor_dice \
    --maximize-best-checkpoint-metric \
    --max-epoch=${max_epoch} \
    --attention-dropout=0.0 \
    --resnet-drop-path-rate=0.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --log-format=simple --log-interval=200 \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --max-image-size=${patch_image_size} \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --n-splits=${n_splits} \
    --fold=${fold} \
    --fp16 \
    --fp16-scale-window=512 \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --det_weight=${det_weight} \
    --cls_weight=${cls_weight} \
    --num-workers=4 > ${log_file} 2>&1

echo "✅ Fold ${fold} training completed!"