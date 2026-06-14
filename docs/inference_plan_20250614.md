# 推理验证计划 & 环境状态 (更新)

**日期**: 2026-06-14
**最后更新**: 2026-06-14 09:40 (GPU 可用，推理链路打通)
**状态**: ✅ GPU 推理链路打通，Dice=0 问题定位中（坐标缩放 bug）

---

## 一、环境现状 (更新)

| 项目 | 状态 | 详情 |
|------|------|------|
| base Python | 3.10.8 | `/root/miniconda3/bin/python3` |
| PyTorch | 2.1.2+cu121 | base 环境 |
| CUDA | 13.0 | nvcc 存在 |
| GPU | **✅ RTX 3090 48GB** | 可用，CUDA 13.0，利用率待优化 |
| 容器内存限制 | 已解除 | 之前 2GB cgroup 已不存在 |
| 系统内存 | 754 GB | |
| RTX 3090 专用内存 | 48 GB | 0MiB 已使用（空闲） |

## 二、依赖修复记录（base 环境）

| 问题 | 修复 | 命令 |
|------|------|------|
| `hydra-core 1.0.7` 与 `omegaconf 2.3.1` 不兼容 | 升级 hydra-core | `pip install "hydra-core==1.3.2"` |
| NumPy `np.float` 等废弃别名 | 脚本首行加 patch | `np.float = float; np.int = int; np.complex = complex; np.bool = bool` |
| fairseq 模块导入链 | `sys.path.insert(0, 'fairseq')` | 本地 fairseq 目录 |
| checkpoint 3.5GB 加载 OOM | mmap 方式加载 | `torch.load(path, mmap=True, map_location='cpu')` |

### 已验证可成功导入的模块

```
from fairseq import tasks, utils, data           ✅
from fairseq.data import Dictionary               ✅
from fairseq.models import ARCH_MODEL_REGISTRY    ✅
from models.polyformer import polyformer          ✅
from tasks.MDC_pretrain import MDCPretrainTask    ✅
from data.MDC_dataset import MedicalPolyDataset   ✅
from bert.tokenization_bert import BertTokenizer  ✅
import torch, nibabel, cv2, numpy                 ✅
```

## 三、关键路径确认

| 路径 | 状态 | 说明 |
|------|------|------|
| `/root/autodl-tmp/MDC/20260512_1311/checkpoint_best.pt` | ✅ 3.53 GB | 训练好的 PolyFormer-B 权重 |
| `/root/autodl-tmp/processed_imagesTr/` | ✅ 281 文件 | 预处理后的 CT 图像 (.nii.gz) |
| `/root/autodl-tmp/processed_labelsTr/` | ✅ 281 文件 | 预处理后的标签 (.nii.gz) |
| `/root/autodl-tmp/MRIpolySeg/fairseq/` | ✅ | 本地 fairseq 源码 |
| `/root/autodl-tmp/pretrained_weights/RadBERT/` | ✅ | RadBERT tokenizer |
| `/root/autodl-tmp/Data/Task07_Pancreas/` | ✅ | 原始 MSD 数据 |
| `/root/autodl-tmp/MRIpolySeg/data/MDC512_new_annotations.p` | ✅ 13 MB | Pickle 标注库 |

## 四、Checkpoint 信息

| 参数 | 值 |
|------|-----|
| 模型架构 | `polyformer_b` |
| 总参数量 | 308.6M |
| 权重 key 数 | 959 |
| fp32 存储大小 | 1.23 GB |
| Config key | `cfg` (type: dict) |
| Task | `MDC_pretrain` |
| `max_image_size` | 512 |
| `patch_image_size` | 512 |
| `num_bins` | 64 |
| `max_tgt_length` | 400 |
| `n_splits` / `fold` | 5 / 0 |
| `encoder_layers` | 6 |
| `decoder_layers` | 6 |
| `vis_encoder_type` | `swin-base` |
| FP16 训练 | 是（`fp16=True`） |

## 五、数据预处理关键参数

训练时 `create_MDC_dataset.py` 中的处理步骤：

```
1. MinMax normalize [0, 1]
2. fliplr(rot90(data, k=1))   → 空间变换
3. 3D Smart Crop 256×256      → 前景中心裁剪，随机抖动±30px
4. 3通道灰度图                 → (Z, 256, 256, 3)
5. 坐标归一化除以 self.img_size=512  → ⚠️ 训练时坐标在 [0, 1]
```

### 数据维度

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 NIfTI | (412, 412, 275) | pancreas_001 为例 |
| 预处理后 | (275, 256, 256, 3) | rot90+fliplr+crop 后 |
| 模型输入 | (1, 3, 256, 256) | 单张切片送入 Swin Encoder |

## 六、推理脚本核心逻辑

### run_inference_vis.py (最新，带可视化)

```
load_model_and_task()
  ├─ torch.load(mmap=True) 读取 checkpoint
  ├─ 从 cfg 重建 src_dict + tgt_dict（仅 4 特殊token + 4096 bin tokens = 4100）
  ├─ 构建 MDCPretrainTask
  ├─ 构建 polyformer_b 模型 (ARCH_MODEL_REGISTRY)
  └─ load_state_dict 加载权重

preprocess_3d_volume(nii_path)
  ├─ MinMax normalize [0,1]
  ├─ fliplr(rot90(data, k=1))
  ├─ Smart Crop 256×256
  └─ 3通道灰度图 (Z, 256, 256, 3)

run_single_slice()
  ├─ 图像 → Swin-B Encoder
  ├─ Prompt + BERT tokenize
  ├─ Encoder forward
  ├─ 自回归 Decoder 解码 (max_len=400)
  │   └─ 每步: cls_head→token类型 + reg_head→坐标(x,y)
  ├─ 剥离坐标 → cv2.fillPoly 渲染
  └─ 返回 pred_mask (256, 256)

3D 重建 & 反向变换
  ├─ 正变换: fliplr(rot90(data, k=1))  (X,Y,Z) → (H,W,Z)
  ├─ 逆变换: rot90(fliplr(slice), k=3)  逐切片在 (H,W) 平面还原
  ├─ 轴映射: reverse结果 (Z, W, H) → transpose(1,2,0) → (X, Y, Z)
  └─ nibabel 保存 .nii.gz
```

## 七、已修复 Bug (2026-06-14)

### 7.1 embed_tokens 尺寸不匹配 (🔴 已修复)

Checkpoint 训练时 vocabulary = **4 特殊 token + 64² bin tokens = 4100**。
run_inference.py 错误加载 BPE dict.txt (50,260) → vocab=54,356 与 checkpoint 不匹配。

**Fix**: 移除 `Dictionary.load(bpe_dict_path)`，只构建 bin tokens。BPE 文本由 BERT tokenizer 独立处理。

### 7.2 Swin Transformer fp16 硬编码 (🔴 已修复)

`swin.py:290`: `attn = self.attn_drop(attn).half()` → CPU fp32 推理时 crash。

**Fix**: 删除 `.half()`。

### 7.3 Decoder fp16 硬编码 (🔴 已修复)

`unify_transformer.py:1336`: `x = self.layernorm_embedding(x.half())` → CPU fp32 推理时 crash。

**Fix**: 删除 `.half()`。

### 7.4 patch_masks=None (🟡 已修复)

**Fix**: 改为 `torch.ones(1, dtype=torch.bool)`

### 7.5 反向变换顺序错误 (🔴 已修复)

`reverse_transform` 中 `rot90∘fliplr` 执行顺序颠倒：应该先 `fliplr` 再 `rot90(k=3)`。

**Fix**: `run_inference_vis.py` 中 `rev_transform` 改为 `fliplr` → `rot90(k=3)`。

### 7.6 transpose 轴映射错误 (🔴 已修复)

`reverse_transform` 输出 `(Z, W, H)` 后 transpose 用 `(2,1,0)` 应改为 `(1,2,0)`：
```
reverse_transform 输出: (Z, W, H)
需要: (X, Y, Z) = (W, H, Z)
所以: result.transpose(1, 2, 0) ✓  不是 (2, 1, 0) ✗
```

**Fix**: `transpose(2,1,0)` → `transpose(1,2,0)`。

## 八、当前问题 (待解决)

### 8.1 模型预测 Dice = 0 🔴

经过上述所有修复后，GT 空间对齐已验证 (100% overlap)，但预测仍与 GT 无重叠。

**已排除的原因**:
- GT 反向变换错误 → ✅ 已修复，GT 100% 对齐
- fp16 类型不匹配 → ✅ 已修复
- embed_tokens 尺寸不匹配 → ✅ 已修复
- 输入全黑 → ✅ 已验证非零

**待排查项** (优先级排序):
1. 坐标缩放因子：训练时坐标除以 `img_size=512` 归一化，推理时 `fillPoly(pred_mask, [poly_pts], 1)` 用 `[w, h] = [256, 256]` 放大。但模型输出的坐标范围究竟是 `[0,1]` 还是 `[0, 0.5]`（归一化除以 512 导致的？）
2. BERT prompt 格式与训练不一致
3. 模型输入 resize：`patch_image_size=512` 时是否内部 resize 输入 256→512

### 8.2 GPU 利用率低

RTX 3090 48GB 空闲，308.6M 参数的模型 fp32 ~1.2GB VRAM，远未塞满。
优化方向：batch inference 一次送入多个切片。

## 九、推理命令 & 速度

```bash
# GPU 全病人推理（275 slices, ~9 分钟)
cd /root/autodl-tmp/MRIpolySeg
python3 run_inference_vis.py --patient pancreas_001

# 快速测试（5 slices from z=80)
python3 run_inference.py --patient pancreas_001 --start-slice 80 --num-slices 5
```

GPU 推理速度: ~0.8-3.5s/slice (快时 w/o CUDA context switch, 慢时含 CUDA overhead)

## 十、输出文件

| 文件 | 说明 |
|------|------|
| `Results/inference_test/pancreas_001_slices/pancreas_001_pred.nii.gz` | 预测 3D mask |
| `Results/inference_test/pancreas_001_slices/pancreas_001_gt.nii.gz` | GT 3D mask |
| `Results/inference_test/pancreas_001_slices/pancreas_001_img.nii.gz` | 预处理后图像 |
| `Results/inference_test/pancreas_001_slices/pancreas_001_per_slice_dice.csv` | 逐切片 Dice |
| `Results/inference_test/pancreas_001_slices/pancreas_001_best_*.png` | Dice 最高 10 切片可视化 |
| `Results/inference_test/pancreas_001_slices/pancreas_001_worst_*.png` | Dice 最低 10 切片可视化 |
