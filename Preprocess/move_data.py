import os
import shutil

# ================= 配置路径 =================
RAW_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/imagesTr" # 原始图像所在的文件夹
PROCESSED_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/processed_imagesTr"  # 已经处理好的文件所在的文件夹 (可以是 nii.gz 也可以是 npy 文件夹)
DEST_DIR = "/root/data/Task07_Pancreas/imagesTr"  # 已经处理过的原图要移去的新文件夹 (例如外部移动硬盘)


# ================= 核心逻辑 =================
def move_processed_raw_files():
    # 确保目标文件夹存在
    os.makedirs(DEST_DIR, exist_ok=True)

    # 1. 获取所有已经处理过的文件的基础名称集合
    print(f"正在扫描已处理文件夹: {PROCESSED_DIR}...")
    processed_basenames = set()

    if not os.path.exists(PROCESSED_DIR):
        print(f"错误: 找不到已处理文件夹 {PROCESSED_DIR}")
        return

    for f in os.listdir(PROCESSED_DIR):
        # 排除 json 配置文件
        if f.endswith('.json'):
            continue

        # 兼容之前我们写过的两种后缀模式：
        # 模式A: 预处理第一阶段的 .nii.gz (如 pancreas_001.nii.gz)
        if f.endswith('.nii.gz'):
            base_name = f.replace('.nii.gz', '')
            processed_basenames.add(base_name)

        # 模式B: 预处理最终阶段的 .npy (如 pancreas_001_img.npy)
        elif f.endswith('_img.npy'):
            base_name = f.replace('_img.npy', '')
            processed_basenames.add(base_name)

        elif f.endswith('_lbl.npy'):
            base_name = f.replace('_lbl.npy', '')
            processed_basenames.add(base_name)

    print(f"-> 发现 {len(processed_basenames)} 个已完成处理的唯一病例 ID。")

    # 2. 遍历原始文件夹，执行移动
    print(f"\n正在扫描原始文件夹: {RAW_DIR} 并执行移动...")
    moved_count = 0

    if not os.path.exists(RAW_DIR):
        print(f"错误: 找不到原始文件夹 {RAW_DIR}")
        return

    for raw_file in os.listdir(RAW_DIR):
        if raw_file.endswith('.nii.gz'):
            raw_base_name = raw_file.replace('.nii.gz', '')

            # 核心判断：如果这个原始文件已经被处理过了
            if raw_base_name in processed_basenames:
                src_path = os.path.join(RAW_DIR, raw_file)
                dst_path = os.path.join(DEST_DIR, raw_file)

                print(f"🚚 移动: {raw_file} -> {DEST_DIR}")

                # 使用 shutil.move 执行剪切操作
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception as e:
                    print(f"移动 {raw_file} 时出错: {e}")

    # 3. 结果统计
    print("\n" + "=" * 50)
    print(f"✅ 清理完成！")
    print(f"✅ 共成功将 {moved_count} 个原始文件移至: {DEST_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    move_processed_raw_files()