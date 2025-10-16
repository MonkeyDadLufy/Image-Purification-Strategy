import os
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import sys

# 标准库导入
from torchvision import transforms, utils
from PIL import Image
from torch.utils import data
from torch import nn, Tensor

# 导入 flow_matching 相关库
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from modules.MDMS import DiffusionUNet


class FFMConfig:
    """配置管理类"""

    def __init__(self, config_path):
        # 使用 UTF-8 编码读取 YAML 文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置设备
        self.device = self.setup_device()

        # 训练配置
        self.lr = self.config['training']['lr']
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.patch_size = self.config['training']['patch_size']

        # 路径配置
        self.train_dir = self.config['training']['train_dir']
        self.val_dir = self.config['training']['val_dir']
        self.model_save_dir = Path(self.config['training']['model_save_dir'])
        self.model_save_name = self.config['training']['model_save_name']

        # 验证配置
        self.val_frequency = self.config['training']['val_frequency']
        self.val_frequency_after_100 = self.config['training']['val_frequency_after_100']
        self.save_frequency = self.config['training']['save_frequency']

        # 推理配置
        self.ode_steps = self.config['inference']['ode_steps']
        self.ode_method = self.config['inference']['ode_method']
        self.ode_rtol = self.config['inference']['ode_rtol']
        self.ode_atol = self.config['inference']['ode_atol']
        self.ode_step_size = self.config['inference']['ode_step_size']
        self.test_model_path = self.config['inference']['test_model_path']
        self.test_lq_dir = self.config['inference']['test_lq_dir']
        self.test_output_dir = self.config['inference']['test_output_dir']

        # 其他配置
        self.cudnn_benchmark = self.config['device']['cudnn_benchmark']
        self.seed = self.config['device']['seed']

        # 数据配置
        self.transform_mean = self.config['data']['transform']['mean']
        self.transform_std = self.config['data']['transform']['std']
        self.dataloader_num_workers = self.config['data']['dataloader']['num_workers']
        self.dataloader_pin_memory = self.config['data']['dataloader']['pin_memory']
        self.dataloader_persistent_workers = self.config['data']['dataloader']['persistent_workers']
        self.dataloader_prefetch_factor = self.config['data']['dataloader']['prefetch_factor']
        self.dataloader_drop_last = self.config['data']['dataloader']['drop_last']

        # Comet ML 配置
        self.comet_config = self.config['comet_ml']

    def setup_device(self):
        """设置设备并检查可用性"""
        if torch.cuda.is_available():
            # 检查可用的 GPU 数量
            num_gpus = torch.cuda.device_count()
            print(f"检测到 {num_gpus} 个 GPU:")

            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")

            # 自动选择设备
            if num_gpus > 1:
                # 如果有多个 GPU，使用第一个可用的
                device = "cuda:0"
            else:
                device = "cuda:0" if num_gpus > 0 else "cpu"
        else:
            device = "cpu"
            print("未检测到 CUDA 设备，使用 CPU")

        print(f"使用设备: {device}")
        return device

    def setup_environment(self):
        """设置训练环境"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        print(f"最终使用设备: {self.device}")


# 自定义数据集 x_0:lq  x_1:gt
class PairedDataset(data.Dataset):
    def __init__(self, data_dir, transform_mean, transform_std):
        self.gt_dir = os.path.join(data_dir, "gt")
        self.lq_dir = os.path.join(data_dir, "lq")

        # 确保目录存在
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"GT目录不存在: {self.gt_dir}")
        if not os.path.exists(self.lq_dir):
            raise FileNotFoundError(f"LQ目录不存在: {self.lq_dir}")

        self.gt_paths = sorted(Path(self.gt_dir).glob("*.png"))
        self.lq_paths = sorted(Path(self.lq_dir).glob("*.png"))

        if len(self.gt_paths) == 0:
            raise FileNotFoundError(f"在 {self.gt_dir} 中未找到 PNG 文件")
        if len(self.lq_paths) == 0:
            raise FileNotFoundError(f"在 {self.lq_dir} 中未找到 PNG 文件")

        # 检查文件名是否匹配
        gt_names = [p.name for p in self.gt_paths]
        lq_names = [p.name for p in self.lq_paths]

        if gt_names != lq_names:
            print("警告: GT和LQ文件名不完全匹配")
            # 使用文件名交集
            common_names = set(gt_names) & set(lq_names)
            self.gt_paths = [p for p in self.gt_paths if p.name in common_names]
            self.lq_paths = [p for p in self.lq_paths if p.name in common_names]
            print(f"使用 {len(self.gt_paths)} 对匹配的图像")

        # 使用更高效的数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std)
        ])

        # 预加载图像路径到内存
        self.filelist = [p.name for p in self.gt_paths]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # 使用更快的图像加载方式
        gt_img = Image.open(self.gt_paths[index]).convert('RGB')
        lq_img = Image.open(self.lq_paths[index]).convert('RGB')
        return {
            'gt': self.transform(gt_img),
            'lq': self.transform(lq_img),
            'filename': self.filelist[index]
        }


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if t.dim() == 0:
            t = t.reshape(1)
        return self.model(x, t, **extras)

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, model, device='cpu'):
    """加载模型"""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded from {path}")
        return model
    raise FileNotFoundError(f"No model found at {path}")


def create_data_loaders(config):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = PairedDataset(config.train_dir, config.transform_mean, config.transform_std)
    val_dataset = PairedDataset(config.val_dir, config.transform_mean, config.transform_std)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 改为 True 以获得更好的训练效果
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.dataloader_persistent_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        drop_last=config.dataloader_drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, config.dataloader_num_workers),
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.dataloader_persistent_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        drop_last=config.dataloader_drop_last
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    return train_loader, val_loader


def initialize_model(config):
    """初始化模型和优化器"""
    # 初始化模型
    vf = DiffusionUNet().to(config.device)

    # 初始化路径和优化器
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(vf.parameters(), lr=config.lr)

    # 创建模型保存目录
    config.model_save_dir.mkdir(parents=True, exist_ok=True)

    return vf, path, optim


def upsample_images(config, lq_images, model):
    """优化后的推理函数"""
    solver = ODESolver(velocity_model=WrappedModel(model))
    T = torch.linspace(0, 1, config.ode_steps).to(config.device)

    # 确保输入图像有批量维度
    if lq_images.dim() == 3:
        lq_images = lq_images.unsqueeze(0)

    return solver.sample(
        time_grid=T,
        x_init=lq_images,
        method=config.ode_method,
        rtol=config.ode_rtol,
        atol=config.ode_atol,
        step_size=config.ode_step_size,
    )


def train_model(config):
    """训练模型"""
    # 初始化模型
    vf, path, optim = initialize_model(config)
    train_loader, val_loader = create_data_loaders(config)

    # 初始化 Comet ML（如果启用）
    if config.comet_config['enabled']:
        try:
            from comet_ml import Experiment
            experiment = Experiment(
                api_key=config.comet_config['api_key'],
                project_name=config.comet_config['project_name']
            )
            print("Comet ML 实验跟踪已启用")
        except Exception as e:
            print(f"Comet ML 初始化失败: {e}")
            experiment = None
    else:
        experiment = None

    print(f"模型参数量: {round(sum(p.numel() for p in vf.parameters() if p.requires_grad) / 1_000_000, 2)}M")

    # 训练循环
    progress_bar = tqdm(range(config.epochs), desc="Training Epochs", position=0, leave=True)
    global_step = 0

    for epoch in progress_bar:
        if experiment:
            experiment.log_current_epoch(epoch)

        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            x_1 = batch['gt'].to(config.device, non_blocking=True)
            x_0 = batch['lq'].to(config.device, non_blocking=True)

            if config.patch_size:
                x_1 = x_1.view(-1, 3, config.patch_size, config.patch_size)
                x_0 = x_0.view(-1, 3, config.patch_size, config.patch_size)

            t = torch.rand(x_1.shape[0]).to(config.device)
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            loss = torch.pow(vf(path_sample.x_t, path_sample.t).float() - path_sample.dx_t.float(), 2).mean()
            loss.backward()
            optim.step()

            if experiment:
                experiment.log_metric("step_train_loss", loss.item(), step=global_step)

            global_step += 1
            epoch_loss += loss.item()

        # 记录平均损失
        avg_loss = epoch_loss / len(train_loader)
        if experiment:
            experiment.log_metric("avg_train_loss", avg_loss, epoch=epoch)

        epoch_time = time.time() - start_time
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{epoch_time:.2f}s'
        })

        # 定期保存模型
        if epoch % config.save_frequency == 0 and epoch != 0:
            model_path = config.model_save_dir / f"{config.model_save_name}_epoch{epoch}.pth"
            save_model(vf, str(model_path))

        # 验证
        should_validate = (
                (epoch < 100 and epoch % config.val_frequency == 0 and epoch != 0) or
                (epoch >= 100 and epoch % config.val_frequency_after_100 == 0)
        )

        if should_validate:
            validate_model(config, vf, val_loader, epoch, experiment)

    # 保存最终模型
    final_model_path = config.model_save_dir / f"{config.model_save_name}_final.pth"
    save_model(vf, str(final_model_path))

    return vf


def validate_model(config, model, val_loader, epoch, experiment=None):
    """验证模型"""
    with torch.no_grad():
        val_start_time = time.time()
        val_loss = 0.0
        val_images = []
        step = 0

        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False, position=1)

        for batch in val_progress_bar:
            x_1 = batch['gt'].to(config.device, non_blocking=True)
            x_0 = batch['lq'].to(config.device, non_blocking=True)

            # 推理
            sr_tensor = upsample_images(config, x_0, model)
            loss = (sr_tensor - x_1).abs().mean()
            val_loss += loss.item()

            # 记录最后一批数据用于可视化
            if step == len(val_loader) - 1:
                val_images = {
                    'lq': (x_0 + 1) * 0.5,
                    'gt': (x_1 + 1) * 0.5,
                    'recon': (sr_tensor + 1) * 0.5
                }

            # 更新进度条
            val_progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{val_loss / (step + 1):.4f}',
                'time': f'{time.time() - val_start_time:.2f}s'
            })

            if experiment:
                experiment.log_metric("step_val_loss", loss.item(), step=step)

            step += 1

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        if experiment:
            experiment.log_metric("avg_val_loss", avg_val_loss, epoch=epoch)

        # 保存最后一组图片
        if val_images:
            utils.save_image(val_images['lq'], str(config.model_save_dir / f'lq-{epoch}.png'), nrow=6)
            utils.save_image(val_images['gt'], str(config.model_save_dir / f'gt-{epoch}.png'), nrow=6)
            utils.save_image(val_images['recon'], str(config.model_save_dir / f'recon-{epoch}.png'), nrow=6)


def test_model(config):
    """测试模型"""
    # 初始化模型
    vf = DiffusionUNet().to(config.device)

    # 加载模型
    load_model(config.test_model_path, vf, config.device)

    # 创建输出目录
    output_dir = Path(config.test_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 批量推理
    lq_files = [f for f in os.listdir(config.test_lq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.transform_mean, std=config.transform_std)
    ])

    print(f"开始处理 {len(lq_files)} 张图像...")

    for filename in tqdm(lq_files, desc="Processing images"):
        lq_path = os.path.join(config.test_lq_dir, filename)

        with torch.no_grad():
            lq_img = Image.open(lq_path).convert('RGB')
            lq_tensor = transform(lq_img).unsqueeze(0).to(config.device)
            sr_tensor = upsample_images(config, lq_tensor, vf)

            # 后处理
            sr_tensor = sr_tensor.squeeze(0).cpu()
            sr_tensor = (sr_tensor * 0.5 + 0.5).clamp(0, 1)

            # 保存结果
            output_path = output_dir / filename
            transforms.ToPILImage()(sr_tensor).save(str(output_path))

    print(f"测试完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FFM 训练和推理脚本')
    parser.add_argument('--config', type=str, default='configs/FFM.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'train_test'],
                        help='运行模式: train, test, 或 train_test')
    parser.add_argument('--model_path', type=str,
                        help='测试时指定的模型路径（可选）')

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return

    try:
        # 加载配置
        config = FFMConfig(args.config)
        config.setup_environment()

        if args.mode in ['train', 'train_test']:
            print("开始训练...")
            trained_model = train_model(config)

        if args.mode in ['test', 'train_test']:
            print("开始测试...")

            # 如果指定了模型路径，则使用指定的路径
            if args.model_path:
                config.test_model_path = args.model_path

            test_model(config)

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()