import os
import glob
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from model.networks import get_nets
from model.losses import get_loss, CVAELoss

# 开启 cuDNN 基准测试，寻找当前固定图像尺寸下最快的卷积算法
torch.backends.cudnn.benchmark = True

class Config:
    TRAIN_BLUR_DIR = r'D:\PRO\Test\demo1\CGdemo2\data\train\snow\\'
    TRAIN_SHARP_DIR = r'D:\PRO\Test\demo1\CGdemo2\data\train\gt\\'
    SAVE_DIR = './checkpoints/'

    EPOCHS = 200
    BATCH_SIZE = 8
    IMG_SIZE = (448, 640)
    LR = 1e-4
    BETA1 = 0.5
    BETA2 = 0.999

    MODEL_DICT = {
        'g_name': 'cvae',
        'd_name': 'patch_gan',
        'd_layers': 3,
        'norm_layer': 'instance',
        'dropout': True,
        'blocks': 9,
        'learn_residual': True,
        'pretrained': True,
        'content_loss': 'perceptual',
        'disc_loss': 'ragan-ls',
        'adv_lambda': 0.01
    }

class SimplePairedDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, size=(448, 640)):
        self.blur_paths = sorted(glob.glob(os.path.join(blur_dir, '*.*')))
        self.sharp_paths = sorted(glob.glob(os.path.join(sharp_dir, '*.*')))

        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return min(len(self.blur_paths), len(self.sharp_paths))

    def __getitem__(self, idx):
        img_blur = Image.open(self.blur_paths[idx]).convert('RGB')
        img_sharp = Image.open(self.sharp_paths[idx]).convert('RGB')
        return self.transform(img_blur), self.transform(img_sharp)

def train():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG, netD = get_nets(cfg.MODEL_DICT)
    netG = netG.to(device)
    if netD is not None:
        netD = netD.to(device)

    # 载入普通 Loss 与专属的 CVAE Loss
    content_loss_fn, disc_loss_fn = get_loss(cfg.MODEL_DICT)
    cvae_loss_fn = CVAELoss(kld_weight=0.0005).to(device)

    optG = torch.optim.Adam(netG.parameters(), lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))
    if netD is not None:
        optD = torch.optim.Adam(netD.parameters(), lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2))

    # 混合精度 (AMP)
    scaler_G = torch.cuda.amp.GradScaler()
    if netD is not None:
        scaler_D = torch.cuda.amp.GradScaler()

    dataset = SimplePairedDataset(cfg.TRAIN_BLUR_DIR, cfg.TRAIN_SHARP_DIR, cfg.IMG_SIZE)

    # 优化数据加载
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=4, pin_memory=True)

    print(f"🚀 开始训练! 模型: {cfg.MODEL_DICT['g_name']}, 分辨率: {cfg.IMG_SIZE}, 数据量: {len(dataset)}")

    for epoch in range(cfg.EPOCHS):
        netG.train()
        if netD is not None:
            netD.train()

        # ==========================================================
        # 💡 KL 退火策略 (KL Annealing):
        # 前 10 个 Epoch 关闭 KLD，让生成器专心学习 U-Net 重构去雪；
        # 第 11 到 50 个 Epoch，KLD 权重从 0 线性增加到目标值，平滑过渡。
        # ==========================================================
        if epoch < 10:
            current_kld_weight = 0.0
        elif epoch < 50:
            current_kld_weight = 0.0005 * ((epoch - 10) / 40.0)
        else:
            current_kld_weight = 0.0001

        cvae_loss_fn.kld_weight = current_kld_weight

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch + 1:03d}/{cfg.EPOCHS:03d}]",
            bar_format="{desc} |{bar:30}| {n_fmt}/{total_fmt} [耗时: {elapsed} < 剩余: {remaining}] {postfix}",
            colour="#00FF00"
        )

        smooth_loss_G = 0.0
        smooth_loss_D = 0.0

        for i, (blur_imgs, sharp_imgs) in progress_bar:
            blur_imgs = blur_imgs.to(device, non_blocking=True)
            sharp_imgs = sharp_imgs.to(device, non_blocking=True)

            # ==========================================================
            # 🔨 A. 区分不同生成器的前向传播逻辑
            # ==========================================================
            with torch.cuda.amp.autocast():
                if cfg.MODEL_DICT['g_name'] == 'cvae':
                    # CVAE训练模式：输入 退化条件(blur_imgs) + 真实目标(sharp_imgs)
                    fake_imgs, mean, logstd = netG(blur_imgs, sharp_imgs)
                else:
                    fake_imgs = netG(blur_imgs)

            # ==========================================================
            # 🔨 B. 优化判别器 D
            # ==========================================================
            if netD is not None:
                optD.zero_grad()
                with torch.cuda.amp.autocast():
                    loss_D = disc_loss_fn.get_loss(netD, fake_imgs.detach(), sharp_imgs)
                scaler_D.scale(loss_D).backward()
                scaler_D.step(optD)
                scaler_D.update()
            else:
                loss_D = torch.tensor(0.0)

            # ==========================================================
            # 🔨 C. 优化生成器 G
            # ==========================================================
            optG.zero_grad()
            with torch.cuda.amp.autocast():
                # 根据模型使用对应的重建 Loss
                if cfg.MODEL_DICT['g_name'] == 'cvae':
                    loss_G_content, _, _ = cvae_loss_fn(fake_imgs, sharp_imgs, mean, logstd)
                else:
                    loss_G_content = content_loss_fn(fake_imgs, sharp_imgs)

                # 添加对抗 Loss
                if netD is not None:
                    loss_G_adv = disc_loss_fn.get_g_loss(netD, fake_imgs, sharp_imgs)
                    loss_G = loss_G_content + cfg.MODEL_DICT['adv_lambda'] * loss_G_adv
                else:
                    loss_G = loss_G_content

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optG)
            scaler_G.update()

            # ==========================================================
            # 📈 计算指数移动平均 (EMA) 并更新面板
            # ==========================================================
            curr_loss_G = loss_G.item()
            curr_loss_D = loss_D.item() if netD is not None else 0.0

            if i == 0:
                smooth_loss_G = curr_loss_G
                smooth_loss_D = curr_loss_D
            else:
                smooth_loss_G = 0.9 * smooth_loss_G + 0.1 * curr_loss_G
                smooth_loss_D = 0.9 * smooth_loss_D + 0.1 * curr_loss_D

            progress_bar.set_postfix({
                'G_loss(平滑)': f"{smooth_loss_G:.4f}",
                'D_loss(平滑)': f"{smooth_loss_D:.4f}"
            })

        # 每 10 个 Epoch 保存一次权重
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), os.path.join(cfg.SAVE_DIR, f"netG_{cfg.MODEL_DICT['g_name']}_ep{epoch + 1}.pth"))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()