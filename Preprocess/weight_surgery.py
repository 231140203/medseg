import torch

print("💉 准备进行模型权重手术...")

# 1. 路径设置 (请确保路径与你服务器上的一致)
polyformer_ckpt_path = "/root/autodl-tmp/pretrained_weights/polyformer_b_pretrain.pt"
radbert_bin_path = "/root/autodl-tmp/pretrained_weights/RadBERT/pytorch_model.bin" # 你刚刚下载的 RadBERT
new_ckpt_path = "/root/autodl-tmp/pretrained_weights/polyformer_radbert_pretrain.pt"

# 2. 加载旧的 PolyFormer 权重
print("-> 正在加载 PolyFormer 权重...")
ckpt = torch.load(polyformer_ckpt_path, map_location="cpu")
poly_state = ckpt["model"]

# 3. 加载新的 RadBERT 权重
print("-> 正在加载 RadBERT 权重...")
radbert_state = torch.load(radbert_bin_path, map_location="cpu")

# 4. 找到旧权重中，文本编码器的前缀 (通常包含 'text_encoder' 或者 'bert')
keys_to_delete = []
text_encoder_prefix = ""

for k in poly_state.keys():
    # 寻找 embedding 层来确定前缀
    if "embeddings.word_embeddings.weight" in k:
        text_encoder_prefix = k.replace("embeddings.word_embeddings.weight", "")
        break

if text_encoder_prefix == "":
    print("❌ 找不到文本编码器的前缀，请检查权重字典！")
    exit()

print(f"-> 侦测到 PolyFormer 中文本编码器的前缀为: '{text_encoder_prefix}'")

# 5. 挖掉旧的 BERT 权重
for k in list(poly_state.keys()):
    if k.startswith(text_encoder_prefix):
        keys_to_delete.append(k)

for k in keys_to_delete:
    del poly_state[k]

print(f"-> 成功切除 {len(keys_to_delete)} 个旧的 BERT 张量。")

# 6. 缝合新的 RadBERT 权重
for k, v in radbert_state.items():
    new_key = text_encoder_prefix + k
    poly_state[new_key] = v

print(f"-> 成功缝合 {len(radbert_state.keys())} 个 RadBERT 张量。")

# 7. 另存为新的预训练文件
torch.save(ckpt, new_ckpt_path)
print(f"✅ 手术圆满成功！新的权重已保存至: {new_ckpt_path}")