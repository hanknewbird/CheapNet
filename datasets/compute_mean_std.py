# coding: utf-8
import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

train_txt_path = os.path.join("DataList.txt")

CNum = 100     # 挑选多少图片进行计算

img_h, img_w = 512, 512  # 缩小为512*512
imgs = np.zeros([img_w, img_h, 1])  # 制定图像格式为w,h,c,b
means, stdevs = 0, 0  # 待存储的均值和方差

with open(train_txt_path, 'r') as f:
    lines = f.readlines()  # 读取DataList内的图像路径,每行为列表中的一项
    random.shuffle(lines)  # 随机打乱

    for i in range(CNum):  # 取出打乱后的图像
        img_path = lines[i].rstrip().split()[0]     # 删除空格后,按照空格切片,取出图像路径

        img = cv2.imread(img_path, 0)               # 以灰度图格式读取图像
        img = cv2.resize(img, (img_h, img_w))       # 将图像长宽缩放至512*512

        img = img[:, :, np.newaxis]                 # 增加新维度
        imgs = np.concatenate((imgs, img), axis=2)  # 拼接图像

        print(f'正在处理第{i}张图像')

imgs = imgs.astype(np.float32)/255.                 # 图像归一化

pixels = imgs[:, :, :].ravel()  # 拉成一行
means = np.mean(pixels)         # 计算每个通道的均值
stdevs = np.std(pixels)         # 计算每个通道的标准差

print(f"normMean = {means}")
print(f"normStd = {stdevs}")
print(f'transforms.Normalize(normMean = {means}, normStd = {stdevs})')


# DAGM_Class6
# normMean = 0.34778258204460144
# normStd = 0.27724799513816833
