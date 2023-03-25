# -*- coding: utf-8 -*-
"""
# @file name  : predict.py
# @brief      : predict demo
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tools.common_tools import get_cheapnet


def img_transform(img_l, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_l: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_l)
    return img_t


def process_img(img_path):
    """
    图像预处理
    :param img_path: 输入图像路径
    :return:
    """
    norm_mean = [0.348]  # DAGM_class6上的均值
    norm_std = [0.278]   # DAGM_class6上的方差

    inference_transform = transforms.Compose([
        # transforms.Resize((512, 512)),              # 裁剪为(512,512)
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    img_l = Image.open(img_path).convert("L")  # 按照指定格式读取图像

    # img --> tensor
    img_t = img_transform(img_l, inference_transform)  # transform
    img_t = img_t.to(device)  # 推至运算设备

    return img_t, img_l


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    group = 128
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_name = f"CheapNet_G{group}.pkl"
    path_state_dict = os.path.join(BASE_DIR, "model", model_name)
    # 加载模型
    cheapNet = get_cheapnet(device=device, groups=group, vis_model=False, path_state_dict=path_state_dict)
    path_img = os.path.join(BASE_DIR, "image", 'img', "1070.PNG")

    # 加载图像
    img_tensor, img_l = process_img(path_img)

    plt.subplot(2, 5, 1)
    plt.imshow(img_l, cmap='gray')
    plt.title(f'original')
    plt.axis('off')

    for index, threshold in enumerate(thresholds):
        # 预测
        with torch.no_grad():
            outputs = cheapNet(img_tensor.unsqueeze(0))
        outputs = (outputs > threshold)
        pre_idx = outputs.squeeze(0).squeeze(0).cpu().numpy()

        pre_img = Image.fromarray(pre_idx)
        # 若文件夹不存在,则创建
        if not os.path.exists(f"image/G{group}"):
            os.makedirs(f"image/G{group}")
        pre_img.save(f"image/G{group}/{threshold}.png")

        # 可视化
        plt.subplot(2, 5, index+2)
        plt.imshow(pre_idx, cmap='gray')
        plt.title(f'{threshold}')
        plt.axis('off')
    plt.show()
