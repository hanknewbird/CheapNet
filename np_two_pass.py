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
            if binary_img[row][col] == 0:
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

    time_1 = time.time()
    binary_img = neighbor_value(binary_img, offsets, False)
    time_2 = time.time()
    cost_1 = time_2 - time_1
    print(f"cost_1:{cost_1}")
    # print(f'第一遍扫描:\n{binary_img}')

    time_3 = time.time()
    binary_img = neighbor_value(binary_img, offsets, True)
    time_4 = time.time()
    cost_2 = time_4 - time_3
    print(f"cost_2:{cost_2}")
    # print(f'第二遍扫描:\n{binary_img}')
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

            # left = x if x <= left else left
            # bottom = y if y >= bottom else bottom

            left = min(x, left)
            bottom = max(y, bottom)
            right = max(x, right)
            top = min(y, top)

        print(f"left, top, right, bottom, area:{left, top, right, bottom, len(points)}")

        # print(points)


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
    import os
    from PIL import Image
    from tools.common_tools import get_cheapnet
    import torchvision.transforms as transforms
    import torch
    import time

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join(BASE_DIR, "image", 'img', "0447.PNG")

    model_name = "CheapNet.pkl"
    path_state_dict = os.path.join(BASE_DIR, "model", model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cheapNet = get_cheapnet(device=device, vis_model=False, path_state_dict=path_state_dict)
    img_tensor, img_l = process_img(path_img)

    with torch.no_grad():
        outputs = cheapNet(img_tensor.unsqueeze(0))
        outputs = (outputs > 0.5)
        pre_idx = np.int8(outputs.squeeze(0).squeeze(0).cpu().numpy())


    # print(f"原始二值图像:\n{pre_idx}")

    # 调用Two Pass算法，计算两遍扫面的结果
    time_tic = time.time()

    for i in range(10):
        binary_img = Two_Pass(pre_idx, NEIGHBOR_HOODS_4)
        binary_img, points_list = reorganize(binary_img)
        get_info(points_list)

    time_toc = time.time()
    cost_time = time_toc - time_tic
    print(f"all time: {cost_time}")


