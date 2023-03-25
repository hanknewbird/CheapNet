# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def mask(img1, n, img2):
    """
    为连通域填色
    :param img1: 原图
    :param n: 连通域
    :param img2: mask图
    :return: 为原图填色后的图
    """
    h, w = img1.shape
    res = np.zeros((h, w, 3), img1.dtype)
    # 生成随机颜色
    random_color = {}
    for c in range(1, n):
        random_color[c] = [0, 0, 255]
    # 为不同的连通域填色
    for i in range(h):
        for j in range(w):
            item = img2[i][j]
            if item == 0:
                pass
            else:
                res[i, j, :] = random_color[item]
    return res


def get_points(stats):
    """
    默认每次只有一个缺陷
    :param stats:
    :return:
    """
    x1 = stats[1, 0]                # x
    y1 = stats[1, 1]                # y
    x2 = stats[1, 0] + stats[1, 2]  # x + w
    y2 = stats[1, 1] + stats[1, 3]  # x + h
    return [x1, y1, x2, y2]


def mask_original(image, mask_img, alpha):
    """
    为原图填充透明mask
    :param image:
    :param points:
    :param alpha:
    :return:
    """
    # cv.rectangle(image, (points[0], points[1]), (points[2], points[3]), color=(0, 0, 255), thickness=2)
    return cv.addWeighted(image, 1, mask_img, alpha, 0)


if __name__ == '__main__':
    # 对图像进行读取，并转换为灰度图像
    original_img = cv.imread('../image/img/0021.png', cv.IMREAD_COLOR)
    label_img = cv.imread('../image/label/0021_label.png', -1)

    # 统计连通域
    count, dst, stats, centroids = cv.connectedComponentsWithStats(label_img, ltype=cv.CV_16U)

    # 得到填色后的mask图像
    mask_img = mask(label_img, count, dst)

    # 通过连通域得到缺陷矩形框坐标
    abnormal_points = get_points(stats)

    #
    mask_original = mask_original(original_img, mask_img, alpha=0.5)

    # 输出每个连通域的面积
    for s in range(1, count):
        print('第 {} 个连通域的面积为：{}'.format(s, stats[s, 4]))

    # 展示结果
    cv.imshow('Origin', original_img)
    cv.imshow('label', label_img)
    cv.imshow('mask', mask_img)
    cv.imshow('mask_original', mask_original)
    cv.waitKey(0)
    cv.destroyAllWindows()
