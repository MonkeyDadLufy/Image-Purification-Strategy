## Imports and init device
from comet_ml import Experiment
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.utils.data import DataLoader
import os
from torchvision import transforms, utils
from PIL import Image
from torch.utils import data
from pathlib import Path
from torch import nn, Tensor
from tqdm import tqdm
import torch
import time
from modules.MDMS import DiffusionUNet  #MDMS

# 设备初始化
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
torch.backends.cudnn.benchmark = True  # 启用cudNN基准优化器
torch.manual_seed(42)

## Training Setup
lr = 0.0001
batch_size = 1
epochs = 200
# hidden_dim = 256  # 适当减小隐藏层维度
patch_size = False

# 初始化模型
vf = DiffusionUNet().to(device)

path = AffineProbPath(scheduler=CondOTScheduler())
optim = torch.optim.Adam(vf.parameters(), lr=lr)

# 模型保存路径
base_path = Path("./saved_models_isp/models_large/models_MDMS")
model_save_path = str(base_path / f"flow_matching_model.pth")
val_save_path = base_path
base_path.mkdir(parents=True, exist_ok=True)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if t.dim() == 0:
            t = t.reshape(1)
        return self.model(x, t, **extras)

## 自定义数据集 x_0:lq  x_1:gt
class PairedDataset(data.Dataset):
    def __init__(self, data_dir):
        self.gt_dir = os.path.join(data_dir, "gt")
        self.lq_dir = os.path.join(data_dir, "lq")
        self.gt_paths = sorted(Path(self.gt_dir).glob("*.png"))
        self.lq_paths = sorted(Path(self.lq_dir).glob("*.png"))
        assert [p.name for p in self.gt_paths] == [p.name for p in self.lq_paths], "文件名不匹配"

        # 使用更高效的数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
def save_model(model, path):
    torch.save(model.state_dict(), path)
    # print(f"Model saved to {path}")


def load_model(path, model, device='cpu'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded from {path}")
        return model
    raise FileNotFoundError(f"No model found at {path}")


##训练Frequency Domain Flow Matching模型
def train():
    experiment = Experiment(api_key="eEDOoiwzjF6OMzqh4LcGkCCUj",
                            project_name="Flow Matching")
    train_dir = r"C:\Users\pytorch\Desktop\Dataset\dataset\test_train"
    val_dir = r"C:\Users\pytorch\Desktop\Dataset\dataset\test_val"
    train_dataset = PairedDataset(train_dir)
    val_dataset = PairedDataset(val_dir)
    print(f"模型参数量: {round(sum(p.numel() for p in vf.parameters() if p.requires_grad) / 1_000_000, 2)}M")
    # 优化后的DataLoader配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        drop_last=True
    )

    # 使用epoch-based训练代替iteration-based
    progress_bar = tqdm(range(epochs), desc="Training Epochs", position=0, leave=False)
    global_step = 0  # 初始化全局步数计数器
    for epoch in progress_bar:
        # experiment.log_current_epoch(epoch)
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            optim.zero_grad(set_to_none=True)  # 更高效的梯度清零
            x_1 = batch['gt'].to(device, non_blocking=True)
            x_0 = batch['lq'].to(device, non_blocking=True)

            if patch_size:  # patch training
                x_1 = x_1.view(-1, 3, patch_size, patch_size)
                x_0 = x_0.view(-1, 3, patch_size, patch_size)

            t = torch.rand(x_1.shape[0]).to(device)
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            loss = torch.pow(vf(path_sample.x_t, path_sample.t).float() - path_sample.dx_t.float(), 2).mean()
            loss.backward()
            optim.step()

            # experiment.log_metric("step_train_loss", loss.item(), step=global_step)
            global_step += 1  # 更新全局步数
            epoch_loss += loss.item()

        # 更新进度条
        avg_loss = epoch_loss / len(train_loader)
        experiment.log_metric("avg_train_loss", avg_loss, epoch=epoch)
        epoch_time = time.time() - start_time
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{epoch_time:.2f}s'
        },refresh=False)

        # 定期保存模型
        if epoch % 20 == 0 and epoch != 0:
            # val_start_time = time.time()
            save_model(vf, model_save_path.replace('.pth', f'_epoch{epoch}.pth'))
        if (epoch < 100 and epoch % 5 ==0 and epoch !=0) or (epoch >100 and epoch % 10 ==0):
            with torch.no_grad():
                # 初始化验证
                val_start_time = time.time()
                val_loss = 0.0
                val_images = []  # 保存最后一组图片用于可视化
                step = 0

                # 使用tqdm包装val_loader，直接迭代数据而非range
                val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}",
                                        leave=False, position=1)

                for batch in val_progress_bar:
                    x_1 = batch['gt'].to(device, non_blocking=True)
                    x_0 = batch['lq'].to(device, non_blocking=True)

                    # 推理
                    sr_tensor = upsample_images(x_0, vf)
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
                    },refresh=False)

                    # experiment.log_metric("step_val_loss", loss.item(), step=step)
                    step += 1

                # 计算平均验证损失
                avg_val_loss = val_loss / len(val_loader)
                experiment.log_metric("avg_val_loss", avg_val_loss,epoch=epoch)

                # 保存最后一组图片
                if val_images:  # 确保有数据
                    utils.save_image(val_images['lq'], str(val_save_path / f'lq-{epoch}.png'), nrow=6)
                    utils.save_image(val_images['gt'], str(val_save_path / f'gt-{epoch}.png'), nrow=6)
                    utils.save_image(val_images['recon'], str(val_save_path / f'recon-{epoch}.png'), nrow=6)

    # 训练完成后保存最终模型
    save_model(vf, model_save_path.replace('.pth', '_final.pth'))


## 优化后的推理函数
def upsample_images(lq_images: Tensor, model: nn.Module, steps=50):  # 减少步数
    solver = ODESolver(velocity_model=WrappedModel(model))
    T = torch.linspace(0, 1, steps).to(device)
    # 确保输入图像有批量维度
    if lq_images.dim() == 3:
        lq_images = lq_images.unsqueeze(0)
    return solver.sample(
        time_grid=T,
        x_init=lq_images,
        method='euler',
        rtol=1e-4,
        atol=1e-5,
        step_size = 0.02,
    )


def infer_and_save(lq_path, model, output_dir="output", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        lq_img = Image.open(lq_path).convert('RGB')
        lq_tensor = transform(lq_img).unsqueeze(0).to(device)
        sr_tensor = upsample_images(lq_tensor, model)

        # 后处理
        sr_tensor = sr_tensor.squeeze(0).cpu()
        sr_tensor = (sr_tensor * 0.5 + 0.5).clamp(0, 1)

        # 保存结果
        output_path = os.path.join(output_dir, os.path.basename(lq_path))
        transforms.ToPILImage()(sr_tensor).save(output_path)



def test():
    model_path = r"saved_models_isp/models_large/models_MDMS/flow_matching_model_final.pth"
    lq_dir = r"/media/D/ym/dataset/test/lq"
    output_dir = "saved_models_isp/models_large/models_MDMS/MDMS_output"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model(model_path, vf, device)

    # 批量推理
    lq_files = [f for f in os.listdir(lq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(lq_files, desc="Processing images"):
        lq_path = os.path.join(lq_dir, filename)
        infer_and_save(lq_path, vf, output_dir, device)
    print(f"enjoy images in {output_dir}")


if __name__ == '__main__':
    train()
    # test()

