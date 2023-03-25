import cv2
import numpy as np

# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1, 0], [0, 0], [1, 0],
             [-1, 1], [0, 1], [1, 1]]


# 记录结果
def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    # 遍历图像上的每个点
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                # .index方法检测字符串中是否包含子字符串 str
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points


# 四领域或八领域判断
def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows - 1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols - 1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img


# binary_img: bg-0, object-255; int
# 第一遍扫描
def Two_Pass(binary_img, neighbor_hoods):
    # 判断是几邻域
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError
    binary_img = neighbor_value(binary_img, offsets, False)
    print(f'第一遍扫描:\n{binary_img}')

    binary_img = neighbor_value(binary_img, offsets, True)
    print(f'第二遍扫描:\n{binary_img}')
    return binary_img


def get_info(points_list):
    point_info = []
    for points in points_list:
        left, top, right, bottom = 0, 0, 0, 0
        # 1.得到x1,y1,x2,y2
        for index, point in enumerate(points):
            if index == 0:
                left, top, right, bottom = point[0], point[1], point[0], point[1]

            x = point[0]
            y = point[1]

            if x <= left:
                left = x

            if y >= bottom:
                bottom = y

            if x >= right:
                right = x

            if y <= top:
                top = y
        print(f"left, top, right, bottom:{left, top, right, bottom}")
        # 00,23
        # 40,63

        print(points)


if __name__ == "__main__":
    # 创建四行七列的矩阵
    binary_img = np.zeros((4, 7), dtype=np.int16)

    # 指定点设置为255(前景)
    index = [[0, 2], [0, 5],
             [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
             [2, 2], [2, 5],
             [3, 1], [3, 2], [3, 4], [3, 5], [3, 6]]
    for i in index:
        binary_img[i[0], i[1]] = np.int16(255)

    import os
    from PIL import Image

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join(BASE_DIR, "..", "image", 'label', "0021_label.PNG")
    np_img = np.int8(Image.open(path_img).convert("L"))

    print(f"原始二值图像:\n{binary_img}")

    # 调用Two Pass算法，计算两遍扫面的结果
    binary_img = Two_Pass(np_img, NEIGHBOR_HOODS_4)

    binary_img, points_list = reorganize(binary_img)

    get_info(points_list)
