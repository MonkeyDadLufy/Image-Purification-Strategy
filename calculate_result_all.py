import os
import cv2
import numpy as np
from scipy.fftpack import dctn
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error


def safe_imread_grayscale(path):
    """安全读取灰度图（确保返回单通道）"""
    try:
        # 方法2：numpy间接读取
        with open(path, 'rb') as f:
            img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img

        # 方法3：用PIL转换
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert('L'))
            return img
        except ImportError:
            pass

        print(f"警告：无法读取图像 {path}")
        return None

    except Exception as e:
        print(f"读取图像 {path} 时发生错误: {str(e)}")
        return None


def calculate_iwssim(img1, img2, data_range=255):
    """
    计算IW-SSIM (Information Weighted Structural Similarity Index)。
    通过局部信息量（边缘/纹理复杂度）加权SSIM，强调重要区域。
    """
    # 计算SSIM的局部图（需skimage >= 0.18）
    ssim_map = ssim(img1, img2, win_size=7, data_range=data_range, full=True)[1]

    # 计算信息权重（使用梯度幅值作为信息量代理）
    grad_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    info_weight = np.sqrt(grad_x ** 2 + grad_y ** 2)
    info_weight = (info_weight - info_weight.min()) / (info_weight.max() - info_weight.min() + 1e-6)

    # 加权平均
    iwssim = np.sum(ssim_map * info_weight) / np.sum(info_weight)
    return iwssim


from skimage.metrics import structural_similarity as ssim


def calculate_msssim(img1, img2, data_range=255, weights=None):
    """
    计算多尺度SSIM (MS-SSIM)。
    默认权重：5个尺度，权重[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]（来自原论文）
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    msssim = 1.0
    for i, w in enumerate(weights):
        # 计算当前尺度的SSIM
        current_ssim = ssim(img1, img2, win_size=7, data_range=data_range)
        msssim *= current_ssim ** w

        # 下采样（高斯模糊 + 降采样）
        if i < len(weights) - 1:
            img1 = cv2.GaussianBlur(img1, (5, 5), 1.5)[::2, ::2]
            img2 = cv2.GaussianBlur(img2, (5, 5), 1.5)[::2, ::2]

    return msssim


def calculate_hdrvdp(img1, img2, ppd=60, hdr_mode=False):
    """
    简化的HDR-VDP计算（近似实现）。
    参数:
        ppd (float): 每度视角像素数（默认60，对应标准观看条件）。
        hdr_mode (bool): 是否启用HDR模式（需图像为线性亮度值）。
    """
    # 转换为浮点型并归一化
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # 计算亮度差异（简化版，省略CSF和掩蔽效应）
    diff = np.abs(img1 - img2)
    if hdr_mode:
        diff = diff ** 0.6  # HDR非线性校正

    # 应用高斯滤波模拟人眼对比敏感函数（CSF）
    sigma = ppd / 6.0  # 模拟CSF带宽
    diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=sigma)

    # 计算差异可见性指数
    vdp = np.mean(diff) * 100  # 转换为百分比
    return vdp

def calculate_gmsd(img1, img2, normalize=True, return_maps=False):
    """
    计算两幅图像的GMSD (Gradient Magnitude Similarity Deviation)指标。

    参数:
        img1, img2 (numpy.ndarray): 输入的两幅图像（需相同尺寸），支持灰度或RGB。
        normalize (bool): 是否将结果归一化到[0,1]范围（原始GMSD范围约为0~0.5）。
        return_maps (bool): 是否返回局部相似性图（用于可视化）。

    返回:
        gmsd (float): 全局GMSD值（越小表示相似性越高）。
        gms_map (numpy.ndarray, optional): 局部相似性图（仅当return_maps=True时返回）。
    """
    # 转换为灰度图（若为RGB）
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 检查图像尺寸
    assert img1.shape == img2.shape, "输入图像尺寸必须相同"

    # 计算梯度幅值（使用Sobel算子）
    sobel_kernel = cv2.FILTER_SCHARR  # 或使用cv2.Sobel(..., ksize=3)
    grad_x1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_mag1 = np.sqrt(grad_x1 ** 2 + grad_y1 ** 2)

    grad_x2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_mag2 = np.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # 计算局部相似性图（GMS）
    T = 0.002  # 论文中的常数，防止除零
    gms_map = (2 * grad_mag1 * grad_mag2 + T) / (grad_mag1 ** 2 + grad_mag2 ** 2 + T)

    # 计算GMSD（标准差）
    gmsd = np.std(gms_map)

    # 可选归一化（将结果映射到[0,1]）
    if normalize:
        gmsd = (gmsd - 0) / (0.5 - 0)  # 原始GMSD值通常在0~0.5之间

    if return_maps:
        return gmsd, gms_map
    else:
        return gmsd


def calculate_metrics_grayscale(img1_path, img2_path):
    """所有指标都使用灰度图计算"""
    # 读取灰度图
    img1 = safe_imread_grayscale(img1_path)
    img2 = safe_imread_grayscale(img2_path)

    # 验证图像
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")

    # 转换为float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算所有指标（全部使用灰度图）
    try:
        metrics = {
            "SSIM": ssim(img1, img2, win_size=7, data_range=255),
            "PSNR": psnr(img1, img2, data_range=255),
            "RMSE": np.sqrt(mean_squared_error(img1, img2)),
            "FSIM": fsim_grayscale(img1, img2),
            "VIF": vif_grayscale(img1, img2),
            "NQM": nqm_grayscale(img1, img2),
            "GMSD": calculate_gmsd(img1, img2),
            # 新增的三个指标
            "IWSSIM": calculate_iwssim(img1, img2),
            "MS_SSIM": calculate_msssim(img1, img2),
            "HDR_VDP": calculate_hdrvdp(img1, img2)
        }
        return metrics
    except Exception as e:
        raise ValueError(f"指标计算失败: {str(e)}")


# 灰度版FSIM实现
def fsim_grayscale(img1, img2):
    """灰度图专用的FSIM计算"""
    # 转换为0-255范围
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)

    # 计算梯度（简化版）
    gx1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)

    gm1 = np.sqrt(gx1 ** 2 + gy1 ** 2)
    gm2 = np.sqrt(gx2 ** 2 + gy2 ** 2)

    # 计算相似度
    c = 0.0026 * 255
    gms = (2 * gm1 * gm2 + c) / (gm1 ** 2 + gm2 ** 2 + c)

    # 相位一致性（简化处理）
    pc = (gm1 + gm2) / 2
    pc = pc / (pc.mean() + 1e-6)

    return np.sum(gms * pc) / np.sum(pc)


# 灰度版VIF实现（简化）
def vif_grayscale(img1, img2, sigma_nsq=0.1):
    """灰度图专用的VIF计算"""
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    # 高斯金字塔（单层简化）
    def gaussian_pyramid(img):
        return gaussian_filter(img, sigma=1)[::2, ::2]

    ref_pyr = gaussian_pyramid(img1)
    dist_pyr = gaussian_pyramid(img2)

    # 计算局部统计
    window = np.ones((3, 3)) / 9.0
    mu1 = cv2.filter2D(ref_pyr, -1, window)
    mu2 = cv2.filter2D(dist_pyr, -1, window)
    sigma1_sq = cv2.filter2D(ref_pyr ** 2, -1, window) - mu1 ** 2
    sigma2_sq = cv2.filter2D(dist_pyr ** 2, -1, window) - mu2 ** 2
    sigma12 = cv2.filter2D(ref_pyr * dist_pyr, -1, window) - mu1 * mu2

    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq[sigma2_sq < 0] = 0

    g = sigma12 / (sigma1_sq + 1e-6)
    sv_sq = sigma2_sq - g * sigma12

    num = np.log2(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
    den = np.log2(1 + sigma1_sq / sigma_nsq)

    return np.sum(num) / (np.sum(den) + 1e-6)


# 灰度版NQM实现（简化）
def nqm_grayscale(img1, img2):
    """灰度图专用的NQM计算"""

    # DCT变换
    def block_dct(img, block_size=8):
        h, w = img.shape
        dct_blocks = np.zeros_like(img)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = img[i:i + block_size, j:j + block_size]
                dct_blocks[i:i + block_size, j:j + block_size] = dctn(block)
        return dct_blocks

    error = img1 - img2
    dct_ref = block_dct(img1)
    dct_err = block_dct(error)

    # 简化计算（省略CSF权重）
    signal_power = np.mean(dct_ref ** 2)
    noise_power = np.mean(dct_err ** 2)

    return 10 * np.log10(signal_power / (noise_power + 1e-6))


def compare_folders(folder1, folder2):
    """比较两个文件夹中的图像（全灰度计算）"""
    files1 = {f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    common_files = sorted(files1 & files2)

    if not common_files:
        print("警告: 没有找到同名图像文件")
        return

    results = {metric: [] for metric in [
        "SSIM", "PSNR", "RMSE", "FSIM", "VIF", "NQM", "GMSD",
        "IWSSIM", "MS_SSIM", "HDR_VDP"
    ]}

    for filename in common_files:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        try:
            metrics = calculate_metrics_grayscale(img1_path, img2_path)
            for key in results:
                results[key].append(metrics[key])
            print(f"处理成功: {filename}")
        except Exception as e:
            print(f"处理 {filename} 失败: {str(e)}")
            continue

    # 打印结果
    print("\n=== 最终结果 ===")
    for metric, values in results.items():
        if values:
            print(f"{metric}: 平均值={np.mean(values):.4f} ± {np.std(values):.4f}")
        else:
            print(f"{metric}: 无有效数据")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", default=r"C:\Users\64617\Desktop\数据集\isp\test\gt_bar", help="参考图像文件夹")
    parser.add_argument("--folder2", default=r"C:\Users\64617\Desktop\模型测试结果\使用gt_bar作为标签\flow matching\MDMS_output", help="待评估图像文件夹")
    args = parser.parse_args()

    compare_folders(args.folder1, args.folder2)