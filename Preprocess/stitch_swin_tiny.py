import torch
import torch.nn.functional as F
import math

print("💉 准备进行极度硬核的 Swin-Tiny 暴力缝合手术...")

# 1. 路径配置 (请根据实际情况修改)
polyformer_ckpt_path = "/root/autodl-tmp/pretrained_weights/polyformer_radbert_pretrain.pt" # 建议用你上次换好 RadBERT 的版本
tiny_ckpt_path = "/root/autodl-tmp/pretrained_weights/swin_tiny_patch4_window7_224.pth"
new_ckpt_path = "/root/autodl-tmp/pretrained_weights/polyformer_tiny_radbert.pt"

print("-> 加载权重中...")
poly_ckpt = torch.load(polyformer_ckpt_path, map_location="cpu")
poly_state = poly_ckpt["model"]

tiny_ckpt = torch.load(tiny_ckpt_path, map_location="cpu")
tiny_state = tiny_ckpt["model"] if "model" in tiny_ckpt else tiny_ckpt

def tile_and_crop(t_tiny, t_base_shape):
    """
    核心暴力缝合函数：将 Tiny 的张量通过 '复制 (Repeat)' 和 '裁剪 (Crop)' 强行匹配 Base 的形状
    """
    if t_tiny.shape == t_base_shape:
        return t_tiny
        
    # 计算每个维度需要复制的次数
    repeats = []
    for dim_base, dim_tiny in zip(t_base_shape, t_tiny.shape):
        repeats.append((dim_base // dim_tiny) + 1)
        
    # 铺瓷砖一样复制
    tiled_tensor = t_tiny.repeat(*repeats)
    
    # 裁剪出需要的确切大小
    slices = tuple(slice(0, dim) for dim in t_base_shape)
    return tiled_tensor[slices]


def interpolate_relative_position_bias(t_tiny, t_base_shape):
    """专门处理窗口大小不同的相对位置编码 (2D -> 3D插值 -> 2D)"""
    # t_tiny 通常是 (169, H_tiny)，t_base_shape 通常是 (529, H_base)
    h_tiny = t_tiny.shape[-1]
    h_base = t_base_shape[-1]

    # 1. 解压回 3D 空间结构 (例如将 169 还原为 13x13)
    w_dim_tiny = int(math.sqrt(t_tiny.shape[0]))
    t_tiny_3d = t_tiny.view(w_dim_tiny, w_dim_tiny, h_tiny)

    # 2. 先在 Head 通道维度上进行铺砖和裁剪
    repeats = (h_base // h_tiny) + 1
    t_aligned_heads = t_tiny_3d.repeat(1, 1, repeats)[:, :, :h_base]  # 变成 (13, 13, h_base)

    # 3. 准备进行空间插值拉伸
    w_dim_base = int(math.sqrt(t_base_shape[0]))
    # interpolate 必须是 (Batch, Channel, H, W) 且数据类型为 float
    t_aligned_heads = t_aligned_heads.permute(2, 0, 1).unsqueeze(0).float()

    # 将 13x13 插值拉伸为 23x23
    t_interpolated = F.interpolate(t_aligned_heads, size=(w_dim_base, w_dim_base), mode='bicubic', align_corners=False)

    # 4. 重新压扁回 2D 形状 (529, h_base)
    t_interpolated = t_interpolated.squeeze(0).permute(1, 2, 0).reshape(-1, h_base)

    # 保持原有的数据类型（通常是 float32 或 float16）
    return t_interpolated.to(t_tiny.dtype)

print("-> 开始缝合视觉骨干 (Swin Image Encoder)...")

prefix = "encoder.embed_images."
replaced_count = 0

for key_base, tensor_base in poly_state.items():
    if not key_base.startswith(prefix):
        continue

    # 🚨 [新增安全锁]: 绝对不能覆盖框架根据图片大小自动生成的 Buffer 变量！
    if "relative_position_index" in key_base or "attn_mask" in key_base:
        continue

    # 去除前缀，得到在 Swin 官方权重中的真实 key
    key_swin = key_base.replace(prefix, "")

    # 处理 Stage 3 的层数不匹配 (Base 18层, Tiny 6层)
    if "layers.2.blocks." in key_swin:
        parts = key_swin.split(".")
        block_idx = int(parts[3])
        tiny_block_idx = block_idx % 6  # 强行循环映射
        parts[3] = str(tiny_block_idx)
        key_tiny = ".".join(parts)
    else:
        key_tiny = key_swin

    if key_tiny not in tiny_state:
        if "norm1" in key_tiny and key_tiny.replace("norm1", "norm") in tiny_state:
            key_tiny = key_tiny.replace("norm1", "norm")

    if key_tiny in tiny_state:
        tensor_tiny = tiny_state[key_tiny]

        # 自动分流处理
        if "relative_position_bias_table" in key_base:
            new_tensor = interpolate_relative_position_bias(tensor_tiny, tensor_base.shape)
        else:
            new_tensor = tile_and_crop(tensor_tiny, tensor_base.shape)

        poly_state[key_base] = new_tensor
        replaced_count += 1
    else:
        print(f"⚠️ 警告: 找不到对应的 Tiny 权重 -> {key_swin}")

print(f"-> 视觉模块缝合完毕！共强行注入并重塑了 {replaced_count} 个张量。")

# 保存科学怪人
torch.save(poly_ckpt, new_ckpt_path)
print(f"✅ 手术圆满成功！新的实验性权重已保存至: {new_ckpt_path}")