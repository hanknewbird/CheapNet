# -*- coding: utf-8 -*-
"""
# @file name  : predict.py
# @brief      : predict demo
"""
import time
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.common_tools import get_cheapnet
import datetime
import json
import os
from pycococreatortools import pycococreatortools


INFO = {
    "description": "DAGM abnormal COCO dataset",
    "url": "https://github.com/",
    "version": "1.0.0",
    "year": 2023,
    "contributor": "hank",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}


LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]


CATEGORIES = [
    {
        'id': 1,
        'name': 'abnormal',
        'supercategory': 'none',
    },
]


def create_dir(path):
    """
    若文件夹不存在,则创建
    :param path: 路径
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


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
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    img_l = Image.open(img_path).convert("L")  # 按照指定格式读取图像

    # img --> tensor
    img_t = img_transform(img_l, inference_transform)  # transform
    img_t = img_t.to(device)  # 推至运算设备

    return img_t, img_l


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 基础路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备

    data_classes = [6]
    groups = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # groups = [64]
    # thresholds = [0.5]

    for data_class in data_classes:
        for threshold in thresholds:
            for group in groups:
                # config
                model_name = f"CheapNet_G{group}.pkl"
                path_state_dict = os.path.join(BASE_DIR, "model", model_name)

                root_path = os.path.join(BASE_DIR, "..", '..', 'Data', 'DAGM', f'Class{data_class}')
                path_imgs = os.path.join(root_path, 'abnormal', 'valid')
                imgs = os.listdir(path_imgs)

                INFO['description'] = f"DAGM predict info"
                coco_output = {
                    "info": INFO,
                    "licenses": LICENSES,
                    "categories": CATEGORIES,
                    "images": [],
                    "annotations": []
                }

                # 全部id默认为1
                image_id = 1
                segmentation_id = 1

                # 加载模型
                cheapNet = get_cheapnet(device=device, groups=group, vis_model=False, path_state_dict=path_state_dict)
                time_tic = time.time()
                for img in imgs:
                    path_img = os.path.join(path_imgs, img)  # 图片路径
                    image = Image.open(path_img).convert("L")  # 灰度图读取
                    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(path_img), image.size)  # 创建images
                    coco_output["images"].append(image_info)  # 追入coco_output
                    category_info = {'id': 1, 'is_crowd': 0}  # 只有1个类

                    # 加载图像
                    img_tensor, img_original = process_img(path_img)

                    # 预测
                    with torch.no_grad():
                        outputs = cheapNet(img_tensor.unsqueeze(0))
                        outputs = (outputs > threshold)

                    pre_idx = np.int8(outputs.squeeze(0).squeeze(0).cpu().numpy())
                    count, dst, stats, centroids = cv.connectedComponentsWithStats(pre_idx, ltype=cv.CV_16U)
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, pre_idx, img_original.size, tolerance=2
                    )
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id += 1
                    image_id += 1

                time_toc = time.time()

                # 计算花费时间
                cost_time = time_toc - time_tic

                print(f"T{threshold}_G{group} time consuming:{cost_time:.2f}s")

                create_dir(f"{root_path}/Annotations/threshold_{threshold}")
                with open(f'{root_path}/Annotations/threshold_{threshold}/predict_G{group}.json', 'w') as output_json_file:
                    json.dump(coco_output, output_json_file)
                    output_json_file.close()
