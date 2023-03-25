# -*- coding: utf-8 -*-
"""
# @file name  : predict.py
# @brief      : predict demo
"""

import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
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
    groups = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for group in groups:

        # config
        # group = 1024
        model_name = f"CheapNet_G{group}.pkl"
        path_state_dict = os.path.join(BASE_DIR, "model", model_name)
        path_fps_log = os.path.join(BASE_DIR, "model", "fps_report", "obj")
        fps_log_name = os.path.join(path_fps_log, f"CheapNet_G{group}_obj.log")
        path_img = os.path.join(BASE_DIR, "image", 'img')
        imgs = os.listdir(path_img)

        # 若文件夹不存在,则创建
        if not os.path.exists(path_fps_log):
            os.makedirs(path_fps_log)

        # test info
        cycle_number = 50

        # 加载模型
        cheapNet = get_cheapnet(device=device, groups=group, vis_model=False, path_state_dict=path_state_dict)

        for img in imgs:
            img_path = os.path.join(path_img, img)
            # 加载图像
            img_tensor, img_l = process_img(img_path)

            # 预测并计时
            with torch.no_grad():
                time_tic = time.time()
                for i in range(cycle_number):
                    outputs = cheapNet(img_tensor.unsqueeze(0))
                    outputs = (outputs > 0.5)
                    pre_idx = np.int8(outputs.squeeze(0).squeeze(0).cpu().numpy())
                    count, dst, stats, centroids = cv.connectedComponentsWithStats(pre_idx, ltype=cv.CV_16U)
                    # print(f"x y w h area info is :{stats[1]}")
                time_toc = time.time()

            # 计算花费时间
            cost_time = time_toc - time_tic
            per_ms = (cost_time/cycle_number)*1000
            fps = (1000/per_ms)

            with open(f"{fps_log_name}", "a") as f:
                f.writelines(f"Done {cycle_number} time "
                             f"consuming: {cost_time:.8f}s, "
                             f"fps: {fps:.5} img/s, "
                             f"times per image: {per_ms:.5} ms/img\n")
                # f.close()

            print(f"Done {cycle_number} time "
                  f"consuming: {cost_time:.8f}s, "
                  f"fps: {fps:.5} img/s, "
                  f"times per image: {per_ms:.5} ms/img")

            # plt.imshow(pre_idx, cmap="gray")
            # plt.show()
