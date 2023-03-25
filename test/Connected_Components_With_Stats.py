# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


def generate_random_color():
    """
    生成随机颜色
    :return: 返回一个长度为3,范围为0~255的列表
    """
    return np.random.randint(0, 256, 3)


def fill_color(img1, n, img2):
    """
    对原始图像填色
    :param img1: 传入图像
    :param n: 连通域数量
    :param img2: 图像掩码
    :return: 填色后的图像
    """
    h, w = img1.shape  # 传入图像高与宽
    res = np.zeros((h, w, 3), img1.dtype)  # 定义一个空的三维矩阵

    # 生成随机颜色
    random_color = {}
    for c in range(1, n):  # 从1开始是因为,背景也算一个连通域
        random_color[c] = generate_random_color()  # 生成随机颜色

    # 为不同的连通域填色:从左上角开始一列一列扫过去
    for i in range(h):  # 一行一行扫描
        for j in range(w):  # 一列一列
            item = img2[i][j]  # 获取原图对应掩码的标号
            if item == 0:  # 当掩码标号为0时.为背景.不处理
                pass  # 跳过
            else:
                res[i, j, :] = random_color[item]  # 填色
    return res


def mark(img, n, stat, cent):
    """

    :param img: 填色后的图像
    :param n: 连通域数量
    :param stat: 连通域信息
    :param cent: 连通域中心坐标
    :return:
    """
    for i in range(1, n):  # 从1开始是因为连通域中计算了背景
        # 绘制中心点
        cv.circle(img, (int(cent[i, 0]), int(cent[i, 1])), 2, (0, 255, 0), -1)
        # 绘制矩形边框
        color = list(map(lambda x: int(x), generate_random_color()))
        cv.rectangle(img,
                     (stat[i, 0], stat[i, 1]),
                     (stat[i, 0] + stat[i, 2], stat[i, 1] + stat[i, 3]),
                     color)
        # 标记数字
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img,
                   str(i),
                   (int(cent[i, 0] + 5), int(cent[i, 1] + 5)),
                   font,
                   0.5,
                   (0, 0, 255),
                   1)


if __name__ == '__main__':
    # 对图像进行读取
    img = cv.imread('./images/label.PNG', -1)

    # 统计连通域,count连通域数量,mask统计后的图,stats每一个标记的统计信息,centroids连通域的中心点
    count, mask, stats, centroids = cv.connectedComponentsWithStats(img, ltype=cv.CV_16U)

    # 为不同的连通域填色
    result = fill_color(img, count, mask)

    # 绘制外接矩形及中心点，并进行标记
    mark(result, count, stats, centroids)

    # 输出每个连通域的面积
    for s in range(1, count):
        print('第 {} 个连通域的面积为：{}'.format(s, stats[s, 4]))

    # 展示结果
    cv.imshow('Origin', img)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
