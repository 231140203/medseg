# 修改报告

## 2026-06-14 #3 — 推理链路打通 & Bug 修复

### 现在用的脚本

**主推理脚本：`run_inference_vis.py`**（带完整可视化 & 逐切片 Dice）
```bash
cd /root/autodl-tmp/MRIpolySeg
python3 run_inference_vis.py --patient pancreas_001
```

**简化推理：`run_inference.py`**（支持 `--start-slice` `--num-slices`）
```bash
python3 run_inference.py --patient pancreas_001 --start-slice 80 --num-slices 5
```

### 修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `run_inference_vis.py` | **新建** (353行) | GPU 推理 + 逐切片 Dice CSV + Top/Bottom PNG 可视化 |
| `run_inference.py` | **修改** | GPU 切换 + start-slice 参数 + embed_tokens/patch_masks 修复 |
| `models/polyformer/swin.py` | **修改** | 删除 L290 `.half()` fp16 硬编码 |
| `models/polyformer/unify_transformer.py` | **修改** | 删除 L1336 `.half()` fp16 硬编码 |
| `docs/inference_plan_20250614.md` | **更新** | 完整环境 & Bug & 修复记录 |
| `modification_report.md` | 本文 | 当前摘要 |

### 已修复 Bug

1. **embed_tokens 尺寸**: checkpoint vocab=4100, 代码错误 load BPE dict → 54360。**Fix:** 只用 bin tokens
2. **Swin fp16 hardcoding**: `attn_drop(attn).half()` CPU fp32 crash。**Fix:** 删 `.half()`
3. **Decoder fp16 hardcoding**: `layernorm_embedding(x.half())`。**Fix:** 删 `.half()`
4. **patch_masks=None**: `~NoneType` crash。**Fix:** `torch.ones(1, dtype=torch.bool)`
5. **reverse_transform 顺序**: `rot90∘fliplr` 顺序错。**Fix:** 改为 `fliplr∘rot90(k=3)`
6. **transpose 轴映射**: `transpose(2,1,0)` 应为 `transpose(1,2,0)`。**Fix:** 改轴序

### 当前状态

✅ GPU 推理 9 分钟跑完 275 slices
✅ GT 空间 100% 对齐
✅ 输入图像非零
✅ 模型输出非零 (~143K voxels)
❌ **Dice=0**: pred 像素和 GT 像素无空间重叠 → 坐标缩放因子 bug（模型坐标 ×256 vs 实际 ×512 偏移）

### 下一步

1. **根因排查**: 检查 `get_predictions_and_masks` 中 `poly_pts = (poly.reshape(-1, 2) * [w, h])` 的 w/h 应该是 256 还是 512
2. **batch inference**: GPU 利用率低，可一次推多个切片
3. **多病人评估**: 扩展到 10+ 病人

---
## 2026-06-14 #2 — 推理链路打通（已废弃旧内容）
...
