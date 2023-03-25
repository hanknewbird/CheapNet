# -*- coding: utf-8 -*-
"""
# @file name  : compute_trainList.py
# @brief      : 得到训练图像列表
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径


if __name__ == '__main__':
    """
    将cifar10解压到Data文件夹后运行本程序
    """

    # 训练集文件夹路径
    train_o_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "Class6", "Train")
    for img_name in os.listdir(train_o_dir):  # 训练集文件夹
        if img_name.endswith(".PNG"):
            img_path = os.path.join(train_o_dir, img_name)
            with open('DataList.txt', 'a') as f:
                f.write(img_path+"\n")
                f.flush()
        else:
            print(f'{img_name}不是图像')

    # 测试集文件夹路径
    train_o_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "Class6", "Test")
    for img_name in os.listdir(train_o_dir):  # 测试集文件夹
        if img_name.endswith(".PNG"):
            img_path = os.path.join(train_o_dir, img_name)
            with open('DataList.txt', 'a') as f:
                f.write(img_path+"\n")
                f.flush()
        else:
            print(f'{img_name}不是图像')
