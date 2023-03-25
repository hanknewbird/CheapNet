# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


def get_edge_point(image_edge, step=10):
    h, w = image_edge.shape
    res = np.zeros((h, w, 3), image_edge.dtype)

    for i in range(0, h, step):
        for j in range(0, w):
            item = image_edge[i][j]
            if item == 0:
                pass
            else:
                res[i, j, :] = [255, 255, 255]
                # print(f'{i},{j}:{image_edge[i][j]}')
    return res


if __name__ == '__main__':
    # 读取图像
    image = cv.imread('./images/label.PNG', -1)

    # 大阈值检测图像边缘
    image_edge = cv.Canny(image, 100, 200, apertureSize=3)

    edge_points = get_edge_point(image_edge, 100)

    # 显示结果
    cv.imshow('Result_high', image_edge)
    cv.imshow('edge', edge_points)
    cv.waitKey(0)
    cv.destroyAllWindows()

