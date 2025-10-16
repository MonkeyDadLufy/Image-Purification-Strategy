import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

T = 0.1
def detect_black_edges(img, black_threshold=10, min_neighbors=1):
    """黑色边缘检测（与v2.0.py保持一致）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, binary = cv2.threshold(gray, black_threshold, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neighbor_count = cv2.filter2D(cleaned, cv2.CV_16U, neighbor_kernel) // 255
    edge_mask = (cleaned == 255) & (neighbor_count >= min_neighbors)

    return edge_mask.astype(np.uint8) * 255


def process_pair(lq_path, gt_path, lq_output_dir, gt_output_dir):
    """处理单对图像并保存结果"""
    lq = cv2.imread(str(lq_path))
    gt = cv2.imread(str(gt_path))

    if lq is None or gt is None:
        print(f"警告：无法读取 {lq_path.name} 或 {gt_path.name}")
        return

    # 执行边缘检测和融合
    lq_edges = detect_black_edges(lq)
    gt_edges = detect_black_edges(gt)
    common_mask = lq_edges + gt_edges

    # 应用融合
    common_mask = common_mask.astype(np.float32) / 255.0
    common_mask = np.expand_dims(common_mask, axis=-1)  # 将形状从 (512, 512) 改为 (512, 512, 1)
    gt_bar = common_mask * lq + (1 - common_mask) * gt
    # lq_bar = common_mask * gt + (1 - common_mask) * lq
    lq_bar = common_mask * gt + (1 - common_mask) * ( lq * (1 - T) +  gt * T )  # 通过self.T让lq_bar有一点gt的纹理，即π1`略微偏向π0 T取值分别为[0.1,0.2,0.3,0.4]

    # 保存结果（保持原始文件名）
    gt_bar_path = os.path.join(gt_output_dir, gt_path.name)
    lq_bar_path = os.path.join(lq_output_dir, lq_path.name)
    cv2.imwrite(gt_bar_path, gt_bar)
    cv2.imwrite(lq_bar_path, lq_bar)


def batch_process(input_dir, output_dir):
    """批量处理目录中的所有图像对"""
    lq_dir = os.path.join(input_dir, "lq")
    gt_dir = os.path.join(input_dir, "gt")

    # 创建输出目录
    lq_output_dir = os.path.join(output_dir, "lq_bar")
    gt_output_dir = os.path.join(output_dir, "gt_bar")
    os.makedirs(lq_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    # 获取配对文件列表
    lq_paths = sorted(Path(lq_dir).glob("*.png"))
    gt_paths = sorted(Path(gt_dir).glob("*.png"))

    # 验证文件名匹配
    assert [p.name for p in lq_paths] == [p.name for p in gt_paths], "LQ和GT文件名不匹配"

    # 处理所有图像对
    for lq_path, gt_path in tqdm(zip(lq_paths, gt_paths), total=len(lq_paths)):
        process_pair(lq_path, gt_path, lq_output_dir, gt_output_dir)


if __name__ == '__main__':
    # 配置路径（请修改为实际路径）
    input_directory = "/media/kun/E23EBCCFE082CC2D/Users/pytorch/Desktop/MyProjects/other models/traindata/test"  # 包含lq和gt子目录的目录
    output_directory = "/media/kun/E23EBCCFE082CC2D/Users/pytorch/Desktop/MyProjects/other models/traindata/isp/test"  # 结果保存目录

    # 执行批量处理
    print("开始批量处理...")
    batch_process(input_directory, output_directory)
    print(f"处理完成！结果已保存到 {output_directory}")