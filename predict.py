import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# 从 model 文件夹引入生成器获取函数
from model.networks import get_generator


# ==========================================
# 推理与模型配置 (直接内置)
# ==========================================
class Config:
    # 图像尺寸必须与训练时保持一致
    IMG_SIZE = (448, 640)

    # 这里的参数必须与您保存权重时的网络结构完全一致，否则无法加载权重
    MODEL_DICT = {
        'g_name': 'cvae',         # <--- 修改点 1：将生成器切换为 cvae
        'd_name': 'patch_gan',
        'd_layers': 3,
        'norm_layer': 'instance',
        'dropout': True,
        'blocks': 9,
        'learn_residual': True,
        'pretrained': True,
        'content_loss': 'perceptual',
        'disc_loss': 'ragan-ls',
        'adv_lambda': 0.001
    }


# ==========================================

def predict(weights_path, test_img_dir, output_dir):
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 实例化生成器
    netG = get_generator(cfg.MODEL_DICT).to(device)

    if not os.path.exists(weights_path):
        print(f"❌ 找不到权重文件: {weights_path}，请先运行 train.py 训练模型！")
        return

    # 2. 安全加载权重
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if 'model' in state_dict:
        state_dict = state_dict['model']

    netG.load_state_dict(state_dict, strict=False)
    netG.eval()

    # 3. 图像预处理 (缩放至 448x640)
    transform = T.Compose([
        T.Resize(cfg.IMG_SIZE),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_paths = sorted(glob.glob(os.path.join(test_img_dir, '*.*')))
    print(f"🚀 找到 {len(img_paths)} 张测试图片，开始推理 (运行设备: {device})...")

    # 4. 开启推理
    with torch.no_grad():
        for path in img_paths:
            img_name = os.path.basename(path)

            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_pil = Image.open(path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 前向传播：由于 CVAEGenerator 在推理时没有传入 target，它会自动从 N(0,1) 采样噪声 z 并解码
            pred_tensor = netG(img_tensor)

            # 后处理映射回普通图片
            pred_numpy = pred_tensor.squeeze().cpu().numpy()
            pred_numpy = np.transpose(pred_numpy, (1, 2, 0))

            pred_numpy = (pred_numpy + 1) / 2.0 * 255.0
            pred_numpy = np.clip(pred_numpy, 0, 255).astype(np.uint8)

            save_path = os.path.join(output_dir, img_name)
            Image.fromarray(pred_numpy).save(save_path)
            print(f"✅ 已保存修复结果: {save_path}")

    print("\n🎉 全部推理完成！请前往输出文件夹查看结果。")

if __name__ == '__main__':

    # <--- 修改点 2：增加了 r 前缀防止转义，并修改为了 CVAE 训练保存的权重名称格式
    WEIGHTS_PATH = r'D:\PRO\Test\demo1\CGdemo2\checkpoints\netG_cvae_ep200.pth'

    # 模糊图片存放的目录 (请确保里面有 .png 或 .jpg 格式的图片)
    TEST_IMG_DIR = r'D:\PRO\Test\demo1\CGdemo2\data\test\snow'

    # 修复后清晰图片的输出目录
    OUTPUT_DIR = r'D:\PRO\Test\demo1\CGdemo2\data\test\out'

    predict(WEIGHTS_PATH, TEST_IMG_DIR, OUTPUT_DIR)