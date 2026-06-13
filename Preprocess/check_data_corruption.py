import os
import numpy as np
from tqdm import tqdm

def scan_dataset_for_corruption(img_dir, lbl_dir, report_file):
    npy_files = [f for f in os.listdir(lbl_dir) if f.endswith('.npy')]
    print(f"🚀 开始进行全量数据体检，共计 {len(npy_files)} 个病人样本...")

    corrupted_registry = {}
    total_corrupted_slices = 0

    for file_name in tqdm(npy_files, desc="扫描进度"):
        img_path = os.path.join(img_dir, file_name)
        lbl_path = os.path.join(lbl_dir, file_name)
        
        # 验证文件对应存在
        if not os.path.exists(img_path):
            print(f"\n⚠️ 警告: 找不到对应的图像文件 {img_path}")
            continue

        try:
            img_volume = np.load(img_path, mmap_mode='r')
            lbl_volume = np.load(lbl_path, mmap_mode='r')
                
        except Exception as e:
            print(f"\n❌ 读取 {file_name} 失败: {e}")
            continue

        bad_slices = []
        for z in range(img_volume.shape[0]):
            img_slice = img_volume[z]
            lbl_slice = lbl_volume[z]
            
            # 核心判断逻辑：图像是否为纯色 (容差 1e-5)
            if abs(img_slice.max() - img_slice.min()) < 1e-3:
                # 检查这张纯色图像上是否依然有 1(胰腺) 或 2(肿瘤) 的标签
                if np.any(lbl_slice > 0):
                    bad_slices.append(z)

        if len(bad_slices) > 0:
            corrupted_registry[file_name] = bad_slices
            total_corrupted_slices += len(bad_slices)

    # ==========================================
    # 生成诊断报告
    # ==========================================
    print("\n" + "="*50)
    print("📊 数据体检报告")
    print("="*50)
    
    if total_corrupted_slices == 0:
        print("✅ 恭喜！未发现图像损坏且包含标签的冲突切片。数据集非常健康！")
    else:
        print(f"🚨 危险警告：在 {len(corrupted_registry)} 个病人中，共发现 {total_corrupted_slices} 张严重冲突的切片！")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"数据体检报告 - 发现 {total_corrupted_slices} 张异常切片\n")
            f.write("="*50 + "\n")
            for pid, slices in corrupted_registry.items():
                msg = f"病人 [{pid}]: 异常切片索引 Z = {slices}"
                print(msg)
                f.write(msg + "\n")
                
        print(f"\n📝 详细报告已保存至: {os.path.abspath(report_file)}")
        print("💡 建议：请去检查你从原始 .nii.gz 转换到 .npy 以及 Crop 时的代码，寻找导致图像变纯色或 Z轴错位的原因。")

if __name__ == "__main__":
    # 请替换为你的实际路径
    IMG_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_imagesTr_512"
    LBL_DIR = "/root/autodl-tmp/Data/Task07_Pancreas/crop_labelsTr_512"
    
    scan_dataset_for_corruption(IMG_DIR, LBL_DIR, "/root/autodl-tmp/Data/Task07_Pancreas/corrupted_samples_report.txt")