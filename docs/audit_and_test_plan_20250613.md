# 数据逻辑全链路审计 & 测试任务清单

**日期**: 2025-06-13  
**项目**: MRIpolySeg — PolyFormer 重构版 MSD 胰腺分割  
**状态**: 审计完成，待 GPU 验证

---

## 第一部分：项目核心文件地图

### 1.1 启动与训练

| 文件 | 作用 | 重要程度 |
|------|------|----------|
| `run_scripts/run_MDC_polyformer_b.sh` | 训练入口 Shell，含全部超参 | ⭐⭐⭐ |
| `train.py` | 训练主循环 + validation + checkpoint 保存 | ⭐⭐⭐ |
| `trainer.py` | Fairseq Trainer（基本未改动） | ⭐⭐ |
| `evaluate.py` | 独立评测脚本入口 | ⭐⭐⭐ |

### 1.2 数据管线

| 文件 | 作用 | 重要程度 |
|------|------|----------|
| `Preprocess/hybrid_preprocessor.py` | 阶段1：NIfTI→预处理NIfTI | ⭐⭐ |
| `data/create_MDC_dataset.py` | 阶段2：NIfTI→.npy体积+Pickle标注库 | ⭐⭐⭐ |
| `data/MDC_dataset.py` | 阶段3：Torch Dataset（核心DataLoader） | ⭐⭐⭐ |
| `data/poly_utils.py` | 多边形工具函数（坐标量化/插值/归一化） | ⭐⭐⭐ |
| `data/base_dataset.py` | Dataset基类（文本预处理/token编码） | ⭐⭐ |
| `data/data_utils.py` | collate/padding/批处理工具 | ⭐⭐ |
| `data/file_dataset.py` | TSV文件读取（已弃用，仅预训练阶段使用） | ⭐ |
| `Preprocess/verify_polygons.py` | 多边形可视化验证 | ⭐ |
| `Preprocess/check_data_corruption.py` | 数据损坏检查 | ⭐ |
| `data/test_dataloader.py` | DataLoader单元测试/可视化 | ⭐⭐ |

### 1.3 任务定义

| 文件 | 作用 | 重要程度 |
|------|------|----------|
| `tasks/base_task.py` | Task基类（词典构建/模型构建/BPE加载） | ⭐⭐⭐ |
| `tasks/MDC_pretrain.py` | **核心Task**：load_dataset/inference/渲染mask/Dice计算 | ⭐⭐⭐ |
| `tasks/refcoco.py` | 原版Refcoco Task（不含mask渲染，仅bbox评测） | ⭐⭐ |

### 1.4 模型

| 文件 | 作用 | 重要程度 |
|------|------|----------|
| `models/polyformer/polyformer.py` | PolyFormer主模型（forward/架构注册） | ⭐⭐⭐ |
| `models/polyformer/unify_transformer.py` | Encoder+Decoder完整实现（1572行） | ⭐⭐⭐ |
| `models/polyformer/swin.py` | Swin Transformer视觉编码器 | ⭐⭐ |
| `models/polyformer/unify_multihead_attention.py` | 多头注意力（含position bias） | ⭐⭐ |
| `models/polyformer/unify_transformer_layer.py` | Transformer层 | ⭐⭐ |
| `models/search.py` | Beam Search | ⭐ |
| `models/sequence_generator.py` | Sequence Generator（本项目未用，手工解码） | ⭐ |

### 1.5 损失函数

| 文件 | 作用 | 重要程度 |
|------|------|----------|
| `criterions/label_smoothed_cross_entropy.py` | 双路Loss(分类CE+回归L1) | ⭐⭐⭐ |
| `criterions/evaluate_predictions.py` | 辅助评测（AP score等） | ⭐⭐ |

### 1.6 配置文件

| 文件 | 作用 |
|------|------|
| `utils/BPE/dict.txt` | GPT-2 BPE词表（50260 tokens） |
| `utils/BPE/encoder.json` | GPT-2 BPE编码器 |
| `utils/BPE/vocab.bpe` | BPE vocab |
| `data/MDC512_new_annotations.p` | **核心标注库**（多边形/Prompt/task） |
| `/root/autodl-tmp/processed_imagesTr/` | **预处理后的.nii.gz图像（281个文件）** |
| `/root/autodl-tmp/processed_labelsTr/` | **预处理后的.nii.gz标签（281个文件）** |
| `/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt` | **训练好的PolyFormer权重** |

---

## 第二部分：全链路数据审计结果

### 2.1 审计通过项 ✅

| # | 检查项 | 详情 |
|---|-------|------|
| 1 | **词典结构** | BPE 50260 + 动态添加 4096 bin tokens = 54356，Token ID = `x_bin*64 + y_bin + 4`，正确 ✅ |
| 2 | **4路双线性融合** | `emb = emb11*dx2*dy2 + emb12*dx2*dy1 + emb21*dx1*dy2 + emb22*dx1*dy1`，原始公式正确 ✅ |
| 3 | **位置编码容量** | `image_bucket_size=42` → 上限 1764 > 1024（256输入32×32特征图），安全 ✅ |
| 4 | **Decoder输出头** | `cls_head`=3分类（COO/SEP/EOS），`reg_head`=MLP→2维+Sigmoid，正确 ✅ |
| 5 | **w/h_resize_ratio=1.0** | 因无resize，恒等映射正确 ✅ |
| 6 | **Collate函数** | src_tokens(BERT padding)、coords(collate_tokens)、images(stack)、labels(stack)，一致 ✅ |
| 7 | **交叉验证** | Patient级 `case_id % 5` 划分 fold，无数据泄漏 ✅ |
| 8 | **Delta计算** | 坐标小数部分正确计算，分离器/终止符处delta置0 ✅ |
| 9 | **GT任务处理** | 肿瘤=标签2，胰腺=标签1，胰腺预测挖掉GT肿瘤区，逻辑合理 ✅ |
| 10 | **region_coord** | 存的是原始像素bbox（非归一化），用于`_inference`中恢复绝对坐标 ✅ |
| 11 | **target_item结构** | `[bbox(2点), poly1各点, [0,0]分隔, poly2各点, [0,0]分隔]`，与Loss index匹配 ✅ |
| 12 | **num_bins一致性** | Dataset和Shell都是64 ✅ |
| 13 | **max_tgt_length** | Shell=400、inference max_len=400、min_len=40，合理 ✅ |
| 14 | **Swin out_index=2** | Stage3 32×32输出，conv_dim=512→proj→768，维度正确 ✅ |

### 2.2 已知问题 ⚠️

#### 问题1：坐标归一化除数 ≠ 渲染乘数

**位置**: `data/MDC_dataset.py:119` + `tasks/MDC_pretrain.py:540`

```python
# Dataset: 坐标归一化用 self.img_size=512（继承自 max_image_size）
poly_scaled = polygon / 512    # 多边形[0~256] → [0~0.5]
target_item = poly_scaled      # 模型学习的目标范围 [0~0.5]

# 渲染: 用实际图像尺寸 256
poly_pts = model_output * [256, 256]  # [0~0.5]×256 → [0~128]
```

**影响**: 预测多边形画在左上角1/4区域，无法覆盖图像右下半部分。 **但训练和推理内部"一致"**（都用了512归一化+256放大），模型可能仍然收敛但Dice很差。

**修复**: 让`self.img_size`等于实际裁剪尺寸256，或渲染时用`max_image_size=512`。

#### 问题2：layernorm_embedding 在 fp16 下对非code token做了 .half()

**位置**: `unify_transformer.py:1335-1336`

```python
if code_masks is None or not code_masks.any() or not getattr(self, "code_layernorm_embedding", False):
    x = self.layernorm_embedding(x.half())  # 强制转为fp16
```

本项目 `code_masks` 始终为 None（数据集不设置这个字段），所以每次都会走这个分支。在 `--fp16` 训练时这是正常的（AMP autocast），但如果单独跑推理（非fp16），这里会精度丢失。

**当前状态**: 训练用 `--fp16` 所以问题不大，推理时需要注意同样开启fp16。

#### 问题3：positioning_transform 定义但未使用

**位置**: `data/MDC_dataset.py:87-88`

```python
self.positioning_transform = T.Compose([
    T.Normalize(mean=[0.5, ...], std=[0.5, ...], max_image_size=max_image_size)
])
# 但在 __getitem__ 中被注释：
# patch_image = self.positioning_transform(patch_image, target=None)
```

**影响**: 图像没有经过 Normalize——模型接收的是 `[0, 1]` 范围的 float32 图像，而非 `[-1, 1]` 的归一化图像。虽然 PolyFormer 原版预训练用的是 Normalize 后的图，但你的模型是从 RadBERT 预训练权重继续训练的，可能已经适应了。影响不确定，值得注意。

#### 问题4：checkpoint restore 路径可能不存在

**位置**: `run_scripts/run_MDC_polyformer_b.sh:28`

```bash
restore_file='/root/autodl-tmp/pretrained_weights/polyformer_radbert_pretrain.pt'
```

但 `/root/autodl-tmp/pretrained_weights/` 目录不存在。当前模型可能是从头训练的，也可能是从其他路径恢复的。checkpoint文件在 `/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt`。

### 2.3 潜在风险 🔴

#### 风险1：坐标归一化的除数差异（问题1的延伸）

即使"内部一致"，这个 bug 仍然让模型的**坐标输出范围只有 `[0, 0.5]`**。因为：
- `F.sigmoid()` 的输出范围是 `[0, 1]`
- 但模型学到的 target 全部在 `[0, 0.5]` 内
- 这意味着**模型回归头的 Sigmoid 输出永远不会用到 `[0.5, 1]` 的范围**
- 浪费了一半的动态范围，影响坐标精度

#### 风险2：`poly_mask` 渲染时的叠加逻辑

**位置**: `tasks/MDC_pretrain.py:530-561`

用了多图层叠加→`>0` 二值化来解决重叠区间镂空问题，但**多个多边形重叠区域会被合并成一块**，对于有确切边界的医学分割任务可能引入伪影。当前 `fix_self_intersection` 被注释，如果模型输出出现自交叉多边形，会导致 `cv2.fillPoly` 产生不可预测的镂空。

---

## 第三部分：测试任务——10个病人分割评测

### 3.1 现状

- **训练好的模型**: `/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt`
- **待测试数据**: `/root/autodl-tmp/processed_imagesTr/*.nii.gz` + `/root/autodl-tmp/processed_labelsTr/*.nii.gz`（281个病人）
- **标注库**: 需要重新生成或直接使用原始`.nii.gz`作为GT对比
- **GPU**: 当前不可用

### 3.2 需要做的事情（按顺序）

```
Phase A: 环境准备 (CPU可行)
├── A1. 确定选哪10个病人（均匀分布、包含有/无肿瘤的）
├── A2. 确认推理所需依赖可以导入（fairseq/torch/Swin/BERT）
├── A3. 准备推理脚本

Phase B: 推理 (需要 GPU)
├── B1. 加载训练好的checkpoint（config + model weights）
├── B2. 重建Task+Model+Dataset
├── B3. 对10个病人的所有有效切片逐一推理
├── B4. 每张切片输出: pred_mask (256×256), GT_mask (256×256), 输入图像
├── B5. 记录每个病人的 Dice/IoU

Phase C: 3D重建与格式转换 (CPU可行)
├── C1. 将每个病人的所有切片按Z轴排序
├── C2. 堆叠为3D volume
├── C3. 反旋转/反裁剪还原到原始物理空间（或直接用原始NIfTI的header）
├── C4. 写入.nii.gz（ITK-SNAP可打开）
├── C5. 对输入图像也做同样的3D stack→.nii.gz

Phase D: 结果汇总
├── D1. 计算每个病人和总体的Dice/IoU
├── D2. 输出对比图（Pred vs GT叠加）
├── D3. 下载到本地用ITK-SNAP查看
```

### 3.3 Phase A 详细计划

#### A1. 选病人规则

```python
# 10个病人，按case_id均匀分布（涵盖不同fold）
# 优先: 有肿瘤的切片 > 无肿瘤的切片（因为肿瘤更稀少、更重要）
# 路径示例: /root/autodl-tmp/processed_imagesTr/pancreas_001.nii.gz
```

建议选择（按case_id均匀采样）:
```
pancreas_001, pancrea_015, pancrea_031, pancrea_047, pancrea_063,
pancreas_079, pancrea_095, pancrea_105, pancrea_120, pancrea_135
```
（实际需要根据文件列表确定）

#### A2. 环境验证

```bash
cd /root/autodl-tmp/MRIpolySeg
source .venv/bin/activate 2>/dev/null || conda activate base
python3 -c "
import torch
from fairseq import tasks, utils
from tasks.MDC_pretrain import MDCPretrainTask
print('OK: all imports work')
"
```

#### A3. 推理脚本核心逻辑

```python
# 伪代码
for patient_id in selected_patients:
    img_nii = sitk.ReadImage(f"processed_imagesTr/{patient_id}.nii.gz")
    lbl_nii = sitk.ReadImage(f"processed_labelsTr/{patient_id}.nii.gz")
    img_arr = sitk.GetArrayFromImage(img_nii)  # (Z, H, W)
    lbl_arr = sitk.GetArrayFromImage(lbl_nii)  # (Z, H, W)
    
    pred_slices = []
    for z in range(img_arr.shape[0]):
        slice_img = preprocess(img_arr[z])     # 裁剪到256×256 + 3通道
        pred_coords = model.inference(slice_img)
        pred_mask = coords_to_mask(pred_coords, 256, 256)
        pred_slices.append(pred_mask)
    
    pred_3d = np.stack(pred_slices, axis=0)
    # 反操作：unflip + unrot90
    pred_3d = np.rot90(np.fliplr(pred_3d), k=-1)  
    # 写入nii.gz
    pred_itk = sitk.GetImageFromArray(pred_3d)
    pred_itk.CopyInformation(img_nii)
    sitk.WriteImage(pred_itk, f"{patient_id}_pred.nii.gz")
```

### 3.4 注意事项

1. **旋转还原**: `create_MDC_dataset.py` 中做了 `np.fliplr(np.rot90(data, k=1))`，3D重建时需要逆向操作：`np.rot90(np.fliplr(data), k=-1)` 或 `np.rot90(np.fliplr(data), k=3)`

2. **裁剪还原**: 推理时对每个病人需要知道裁剪参数(`y_start, x_start`)，否则3D重建的mask放不回原始图像空间。当前标注库不存在（`.p`文件缺失），需要：
   - 方案a: 直接从.full NIfTI生成GT（跳过.p文件），在原始空间评测
   - 方案b: 重新运行`create_MDC_dataset.py`生成.p文件

3. **256 vs 512**: 如果用当前模型推理，输入图像需要裁剪到256×256（与训练一致）。如果用原始NIfTI直接推理，需要先做相同的裁剪+旋转预处理。

4. **GPU显存**: Swin-B模型约200M参数，单张256×256切片推理约需2-3GB显存。

### 3.5 推荐推理命令框架

```bash
# 推理主脚本: run_inference.py (待编写)
cd /root/autodl-tmp/MRIpolySeg
python3 run_inference.py \
    --checkpoint /root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt \
    --data-dir /root/autodl-tmp/processed_imagesTr \
    --label-dir /root/autodl-tmp/processed_labelsTr \
    --output-dir /root/autodl-tmp/MRIpolySeg/Results/inference_test \
    --num-patients 10 \
    --gpu 0
```

---

## 第四部分：整体数据流总结

```
原始MSD Pancreas .nii.gz (Z×H×W, 可变尺寸, CT HU值)
    │ hybrid_preprocessor.py (nnUNet+Dual-Task-Seg混合)
    ├─ 裁除非零区域
    ├─ CT窗位[-100,240]→ MinMax归一化
    └─ 重采样到(1.0, 0.8, 0.8)mm spacing
    ▼
processed_*Tr/*.nii.gz (统一spacing, [0,1]范围)
    │ create_MDC_dataset.py
    ├─ 旋转+翻转对齐
    ├─ 3D Smart Crop (256×256, 病人级共享)
    ├─ cv2.findContours提取轮廓
    ├─ 等距重采样(胰腺64点, 肿瘤32点)
    └─ 生成3句多样化Prompt
    ▼
crop_*Tr/*.npy + MDC_annotations.p
    │ MedicalPolyDataset.__getitem__
    ├─ mmap读取slice → (3,256,256) tensor
    ├─ 多边形坐标÷512归一化 → [0,1]范围
    ├─ 量化→4路token序列+delta插值
    └─ BERT tokenize Prompt
    ▼
Model Input: {patch_images, src_tokens, prev_output_tokens_11/12/21/22, delta_x1/y1/x2/y2}
    │ Encoder: Swin-B(32×32 conv dim=512) → proj(768) + BERT(768) → concat → 12层Transformer
    │ Decoder: 4路token embedding → 双线性融合 → 12层Transformer
    │ Output: cls_head(3类) + reg_head(2维 + sigmoid)
    │ Loss: L1(reg) + CELoss(cls)
    ▼
Validation: 自回归解码 → 剥离bbox → 按SEP切分多边形
    │ fillPoly渲染 → (256,256) mask
    │ 胰腺: 挖掉GT肿瘤区 / 肿瘤: 只取GT class2
    ▼
Dice/IoU vs GT mask
```

---

## 第五部分：下一步行动（需要GPU时执行）

- [ ] 执行第四节3.3的测试任务（选10病人推理）
- [ ] 修复坐标归一化问题（问题1）
- [ ] 修复 positioning_transform 未调用（问题3）
- [ ] 验证问题1修复后Dice是否有改善
- [ ] 生成3D nii.gz供ITK-SNAP可视化

---

## 第六部分：模型架构说明

### 6.1 总览

本项目基于 **PolyFormer**（OFA 的多边形预测变体），将医学影像分割转化为 **Seq2Seq 多边形顶点预测** 问题。模型接收一张 CT 切片和一句文本 Prompt，自回归输出多边形顶点坐标序列，最后用 `cv2.fillPoly` 渲染为分割掩码。

### 6.2 模型不是 nnU-Net

**nnU-Net 只出现在预处理管线中，不出现在模型架构中。**

| 组件 | 来源 | 说明 |
|------|------|------|
| 预处理裁剪 `crop_to_nonzero` | nnU-Net 风格 | 借鉴了 nnU-Net 的 BBox 裁剪逻辑，用于去除 CT 体积中的纯黑背景 |
| 预处理重采样 `resample_3d` | Dual-Task-Seg 风格 | 统一物理分辨率到 `(1.0, 0.8, 0.8)` mm |
| 模型架构 | **PolyFormer (基于 OFA/Swin)** | 与 nnU-Net 完全无关 |

nnU-Net 是纯 CNN 的 U-Net 变体（编码器-解码器结构，跳跃连接），做端到端像素级分割。本项目是 **Transformer 架构**，输出多边形坐标而非像素。两者唯一的交集是预处理阶段借用了 nnU-Net 的数据清洗思路。

### 6.3 视觉编码器：Swin Transformer

**实现文件**: `models/polyformer/swin.py` + `unify_transformer.py:452-476`

```python
# unify_transformer.py:454-456
self.embed_images = SwinTransformer(
    pretrain_img_size=384, window_size=12, embed_dim=128,
    out_indices=[out_index], depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]
)
```

Swin Transformer 是微软研究院 2021 年提出的视觉 Transformer 变体，核心创新是 **Shifted Window Self-Attention**——将特征图划分为固定大小的窗口，在窗口内做自注意力，跨窗口通过"偏移窗口"机制交换信息。

| 特性 | 说明 |
|------|------|
| 版本 | Swin-Base (参数量 ~88M) |
| 窗口大小 | 12×12 |
| 层次结构 | 4 个 Stage，各 Stage 做 2×2 patch merging 降采样 |
| 输出 | `out_index=2` 时取 Stage 3 输出，特征图尺寸为输入的 1/8 |
| 预训练 | ImageNet-22K (`swin_base_patch4_window12_384_22k.pth`) |
| 输入尺寸兼容 | 全卷积设计，不依赖固定输入尺寸。256×256 输入 → Stage3 输出 32×32 |

**Swin 和 Vision Transformer (ViT) 的区别**:
- ViT: 全局自注意力，计算量为 O(N²)，N = 图像 patch 数
- Swin: 窗口内局部自注意力 + 层次化下采样，计算量为 O(N)，更适合高分辨率医学图像

### 6.4 文本编码器：RadBERT（BERT 变体）

**实现文件**: `data/MDC_dataset.py:91` + `unify_transformer.py:624-625`

```python
# Tokenizer
self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/pretrained_weights/RadBERT/')

# Encoder 中的 BERT embedding
token_embedding = self.bert(src_tokens, attention_mask=att_masks)[0]
```

RadBERT 是放射学领域预训练的 BERT 模型，在大量放射学报告文本上做过领域自适应预训练。你的 Prompt 文本（如 `"segment an elongated pancreas containing a round tumor"`）通过 RadBERT tokenizer 分词后，经 BERT embedding 层转换为 token embedding，再与图像特征拼接送入跨模态 Transformer。

| 特性 | 说明 |
|------|------|
| 架构 | BERT-Base (12层, 768维, 12头) |
| Tokenizer | RadBERT 医学词表 |
| 作用 | 将语义 Prompt 编码为稠密向量，引导视觉解码器定位目标器官 |
| 是否 fine-tune | 是，BERT 参数随模型一起训练（未冻结） |

### 6.5 跨模态融合：Transformer Encoder

**实现文件**: `unify_transformer.py:656-814`

```
Image Tokens (1024, 768)  +  Text Tokens (~20, 768)
         │                          │
         └──────── concat ──────────┘
                      │
         ┌───────────▼────────────┐
         │  Transformer Encoder    │
         │  12 layers, 16 heads    │
         │  embed_dim = 768        │
         │  ffn_dim = 4096         │
         │  + Absolute Position    │
         │    Bias (from Swin +    │
         │    BERT position IDs)   │
         │  + Relative Position    │
         │    Bias (token & image) │
         └───────────┬────────────┘
                     ▼
              Encoder Output
```

### 6.6 坐标解码器：4路双线性量化

**实现文件**: `unify_transformer.py:1130-1411`

这是 PolyFormer 的核心创新：

```
Encoder Output
      │
      ▼
┌────────────────────────────────────────────┐
│  4路并行 Token Embedding                    │
│  prev_output_tokens_11 (x_floor, y_floor)  │
│  prev_output_tokens_12 (x_floor, y_ceil)   │
│  prev_output_tokens_21 (x_ceil,  y_floor)  │
│  prev_output_tokens_22 (x_ceil,  y_ceil)   │
│                                              │
│  双线性融合:                                  │
│  emb = emb11·dx2·dy2 + emb12·dx2·dy1        │
│      + emb21·dx1·dy2 + emb22·dx1·dy1        │
└──────────────────┬─────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  Transformer Decoder (12 layers, 16 heads)   │
│  自回归: 每步预测下一个 token                 │
└──────────────────┬──────────────────────────┘
                   ▼
    ┌──────────────┴──────────────┐
    │                              │
    ▼                              ▼
 cls_head (3类)               reg_head (2维)
 Linear(768→3)                MLP(768→768→2)
 token类型预测                 坐标预测 (x,y)
 (COO=0/SEP=1/EOS=2)          F.sigmoid → [0,1]
```

4路量化 + delta 插值的好处：模型可以输出 sub-bin 精度的坐标。例如 `num_bins=64` 时，floor 和 ceil 之间被 delta 插值填补，等效精度远高于 64 级离散化。

### 6.7 完整数据流

```
输入: CT切片(3,256,256) + Prompt文本("segment an elongated pancreas...")

┌─ 视觉分支 ──────────────────────────┐
│ Swin-Base (ImageNet-22K 预训练)      │
│ ├─ Stage1: 256×256 → 128×128       │
│ ├─ Stage2: 128×128 → 64×64         │
│ └─ Stage3: 64×64 → 32×32           │  (out_index=2, 到此停止)
│ conv_dim = 512                       │
│ ↓ Linear(512→768) proj              │
│ Image Tokens: (1024, 768)            │
└─────────────────────────────────────┘
         │
         ├── Concat ───────────────┐
         │                         │
┌─ 文本分支 ───────────────────┐   │
│ RadBERT Tokenizer            │   │
│ ↓                            │   │
│ BERT Embedding (768维)       │   │
│ Text Tokens: (~20, 768)      │   │
└──────────────────────────────┘   │
                                   ▼
         ┌──────────────────────────────────┐
         │ Transformer Encoder (12层, 768维) │
         │ 图像+文本 跨模态注意力融合          │
         │ + 绝对/相对位置偏置                 │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────────┐
         │ Transformer Decoder (12层, 自回归) │
         │ 4路坐标 token → 双线性融合          │
         │ → cls_head: token类型(COO/SEP/EOS)│
         │ → reg_head: 连续坐标(x,y)∈[0,1]   │
         └──────────────┬───────────────────┘
                        ▼
         自回归循环 (max_len=400)
         → 多边形顶点序列
         → cv2.fillPoly → (256,256) 分割Mask
```

### 6.8 与其他框架的对比

| 维度 | nnU-Net | 原始 PolyFormer (Refcoco) | **本项目 (MRIpolySeg)** |
|------|---------|--------------------------|------------------------|
| 视觉骨干 | CNN (U-Net) | Swin-B/L | **Swin-B** |
| 文本骨干 | 无 | BERT-Base | **RadBERT** (医学领域) |
| 输出形式 | 像素级 Mask | Bbox 坐标 | **多边形顶点 → Mask** |
| 任务 | 纯分割 | 指代表达理解 | **医学多类分割** |
| 数据格式 | .nii.gz | TSV+base64 | **.npy Volume + Pickle** |
| 图像输入 | 原始分辨率 | 512×512 Resize | **256×256 Smart Crop** |
| 多类支持 | 原生支持 | 无 | **胰腺+肿瘤双任务** |

### 6.9 关键设计决策

1. **为什么用多边形而非直接分割**：Seq2Seq 多边形预测比像素级分割更稀疏（64个点 vs 65536个像素），Transformer 的序列长度可控，且多边形天然保证分割区域的连通性。

2. **为什么 Crops 到 256 而非原版 512**：减少 Image Tokens 数量（1024 vs 4096），降低 Transformer 计算和显存开销。Swin 的全卷积特性保证了不同输入尺寸的兼容性。

3. **为什么用 RadBERT 而非 BERT-Base**：医学 Prompt 含有领域特定术语（如 "pancreatic tissue", "lesion"），RadBERT 在放射学文本上预训练过，对这些术语的语义理解更好。

4. **为什么 4 路双线性量化**：原版 PolyFormer 的设计。相比单路离散 token，4 路量化可以让连续坐标值通过 delta 插值恢复亚 bin 精度，预测的坐标更精确。

---

*文档生成时间: 2025-06-13*  
*分析工具: Claude Code 静态代码审查*  
*后续更新: 待GPU可用后补充推理结果*
