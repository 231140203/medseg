# 推理验证计划 & 环境状态

**日期**: 2026-06-14
**状态**: 环境已打通全部导入，推理脚本已编写完成，等待 GPU/内存资源后执行

---

## 一、环境现状

| 项目 | 状态 | 详情 |
|------|------|------|
| base Python | 3.10.8 | `/root/miniconda3/bin/python3` |
| PyTorch | 2.1.2+cu121 | base 环境 |
| CUDA | 12.1 | nvcc 存在 |
| GPU | **不可用** | 无 `/dev/nvidia*`，`nvidia-smi` 无输出 |
| 容器内存限制 | **2GB cgroup** | `memory.max = 2147483648`（read-only，无法修改） |
| 系统内存 | 754 GB | 但 cgroup 限制覆盖 |
| polyformer env | Python 3.8 裸壳 | 未安装任何 ML 包（用不上） |

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
| 总参数量 | 320.5M |
| 权重 key 数 | 959 |
| fp32 存储大小 | 1.23 GB |
| Config key | `cfg` (type: dict) |
| Task | `MDC_pretrain` |
| `max_image_size` | 512 |
| `patch_image_size` | 512 （模型内部 resize，输入 256 会被放大到 512） |
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
5. 坐标归一化除以 self.img_size=512  → ⚠️ BUG（应为256）
```

### 数据维度

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 NIfTI | (412, 412, 275) | pancreas_001 为例 |
| 预处理后 | (275, 256, 256, 3) | rot90+fliplr+crop 后 |
| 模型输入 | (1, 3, 256, 256) | 单张切片送入 Swin Encoder |
| 模型内部 resize | 256→512 | `patch_image_size=512` 时自动 resize |
| 输出 mask | (256, 256) | 用 `max_image_size=512` 做 fillPoly |

## 六、推理脚本核心逻辑 (`run_inference.py`)

```
load_model_and_task()
  ├─ torch.load(mmap=True) 读取 checkpoint
  ├─ 从 cfg 重建 src_dict + tgt_dict（BPE + 4096 bin tokens）
  ├─ 构建 MDCPretrainTask
  ├─ 构建 polyformer_b 模型 (ARCH_MODEL_REGISTRY)
  └─ load_state_dict 加载权重

preprocess_3d_volume(nii_path)
  ├─ MinMax normalize [0,1]
  ├─ fliplr(rot90(data, k=1))
  ├─ Smart Crop 256×256
  └─ 3通道灰度图 (Z, 256, 256, 3)

run_inference_single_slice()
  ├─ 图像 → Swin-B Encoder
  ├─ Prompt + BERT tokenize
  ├─ Encoder forward
  ├─ 自回归 Decoder 解码 (max_len=400)
  │   └─ 每步: cls_head→token类型 + reg_head→坐标(x,y)
  ├─ 剥离坐标 → cv2.fillPoly 渲染
  └─ 返回 pred_mask (256, 256)

run_inference_on_patient()
  ├─ 逐切片推理
  ├─ 3D stack → 反裁剪 → 反旋转
  └─ nibabel 保存 .nii.gz
```

## 七、已知问题（之前审计发现，尚未修复）

| # | 问题 | 严重度 | 对推理的影响 |
|---|------|--------|-------------|
| 1 | 坐标归一化用512而非常256 | 🔴 严重 | 模型预测坐标范围 `[0, 0.5]`，fillPoly 时用 `max_image_size=512` 放大到 `[0, 256]`，训练和推理一致，但浪费一半动态范围 |
| 2 | `positioning_transform` 未调用 | 🟡 中 | 图像未做 Normalize（range 保持 `[0,1]` 而非 `[-1,1]`），模型已适应 |
| 3 | `layernorm_embedding` fp16 问题 | 🟡 中 | CPU fp32 推理时不受影响 |
| 4 | Checkpoint restore 路径不存在 | 🟢 低 | 不影响推理 |

## 八、开 GPU 后要做的第一步

### 启动命令
```bash
cd /root/autodl-tmp/MRIpolySeg
python3 run_inference.py --patient pancreas_001 --num-slices 1
```

预期：
- 成功加载模型到 GPU
- 跑 1 个切片全链路（~1-2 秒）
- 输出 pred_mask 和 Dice score
- 确认旋转/裁剪反向变换正确

### 如果这个成功，下一步
```bash
# 全病人推理
python3 run_inference.py --patient pancreas_001

# 10 个病人
python3 run_inference.py --patients \
  pancreas_001 pancreas_032 pancreas_063 pancreas_094 pancreas_125 \
  pancreas_156 pancreas_187 pancreas_218 pancreas_249 pancreas_280
```
