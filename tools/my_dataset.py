# -*- coding: utf-8 -*-
"""
# @file name  : my_dataset.py
# @brief      : 数据集Dataset定义
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class DAGMDataset(Dataset):
    def __init__(self, data_dir, mode='train', img_transform=None, label_transform=None):
        assert (os.path.exists(data_dir)), f"data_dir:{data_dir} 不存在！"

        self.data_dir = data_dir
        self.mode = mode
        self._get_img_info()
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_path, label_path = self.img_info[index]

        img = Image.open(img_path).convert("L")  # L为8比特,灰度图
        img = self.img_transform(img)

        label = Image.open(label_path).convert("1")  # 1为1比特,二值图
        label = self.label_transform(label)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        """
        获取图像路径和目标图像路径
        :return:
        """

        abnormal_dir = os.path.join(self.data_dir, 'abnormal', self.mode)
        label_dir = os.path.join(self.data_dir, 'abnormal_label', self.mode)

        label_paths = os.listdir(label_dir)

        self.img_info = []

        # img, label
        path_img = [(os.path.join(abnormal_dir, i.replace("_label", "")),
                     os.path.join(label_dir, i)) for i in label_paths if i.endswith("PNG")]

        self.img_info.extend(path_img)






























