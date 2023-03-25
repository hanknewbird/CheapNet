import os
import shutil

"""构建数据集"""


def create_dir(path):
    """
    若文件夹不存在,则创建
    :param path: 路径
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def move_img_to_dir(father_dir, imgs, target_dir, target_dir2=""):
    """
    指定传入列表将其转入指定路径
    :param father_dir: 输入图像的文件夹
    :param imgs: 图片文件名
    :param target_dir: 转移路径
    :param target_dir2: 转移路径2
    :return:
    """
    # 1.先创建输出文件夹
    create_dir(target_dir)

    if len(target_dir2) > 2:
        create_dir(target_dir2)

    for _, img in enumerate(imgs):
        # 2.计算目标路径
        input_dir = os.path.join(father_dir, img)

        # 3.转移
        shutil.move(input_dir, target_dir)  # 转移缺陷标签
        print(f"{input_dir}-->{target_dir}")

        if len(target_dir2) > 2:
            abnormal_img_name = img.replace("_label", "")
            abnormal_path = os.path.join(father_dir, '..', abnormal_img_name)

            shutil.move(abnormal_path, target_dir2)  # 转移缺陷标签
            print(f"{abnormal_path}-->{target_dir2}")


def build(data_class, rate, save_normal=False):
    dataset_class = data_class  # 类名
    train_valid_rate = rate  # 训练集比例
    dateset_path = os.path.join('..', '..', '..', 'Data', dataset_class)  # 数据集位置
    output_path = os.path.join('..', '..', '..', 'Data', 'DAGM', dataset_class)  # 输出路径
    input_train_path = os.path.join(dateset_path, 'Train')  # 训练集路径
    input_valid_path = os.path.join(dateset_path, 'Test')  # 验证集路径

    create_dir(output_path)  # 创建文件夹

    input_paths = [input_train_path, input_valid_path]  # 将训练集和验证集保存为数组,方便循环

    flag = 0  # 记录算了多少个文件夹

    for input_path in input_paths:  # 循环2部分组成的数据集

        label_path = os.path.join(input_path, 'Label')  # 缺陷图片文件夹
        imgs_path = []  # 包含缺陷的图片

        for i, img_name in enumerate(os.listdir(label_path)):  # 判断是否为图片
            if img_name.endswith('.PNG'):  # 判断是否为png
                imgs_path.append(img_name)

        imgs_number = len(imgs_path)
        split_number = int(imgs_number * train_valid_rate)

        train_imgs_path = []  # 训练集路径
        valid_imgs_path = []  # 验证集路径

        # 训练集构造
        for i in range(0, split_number):
            train_imgs_path.append(imgs_path[i])

        # 验证集构造
        for i in range(split_number, imgs_number):
            valid_imgs_path.append(imgs_path[i])

        # 移动train
        abnormalLabel_train_dir = os.path.join(output_path, 'abnormal_label', 'train')
        abnormal_train_dir = os.path.join(output_path, 'abnormal', 'train')
        move_img_to_dir(label_path, train_imgs_path, abnormalLabel_train_dir, abnormal_train_dir)

        # 移动valid
        abnormalLabel_valid_dir = os.path.join(output_path, 'abnormal_label', 'valid')
        abnormal_valid_dir = os.path.join(output_path, 'abnormal', 'valid')
        move_img_to_dir(label_path, valid_imgs_path, abnormalLabel_valid_dir, abnormal_valid_dir)

        if save_normal:
            # 处理正常图片
            input_normal_name = []
            for i, name in enumerate(os.listdir(input_path)):
                if name.endswith('.PNG'):  # 判断是否为png
                    input_normal_name.append(name)

            normal_number = len(input_normal_name)
            split_number = int(normal_number * train_valid_rate)

            train_normal = []
            valid_normal = []

            # normal_train
            for i in range(0, split_number):
                train_normal.append(input_normal_name[i])

            # normal_valid
            for i in range(split_number, normal_number):
                valid_normal.append(input_normal_name[i])

            # normal_train输出路径
            target_normal_train_dir = os.path.join(output_path, 'normal', 'train')
            move_img_to_dir(input_path, train_normal, target_normal_train_dir)

            # normal_valid输出路径
            target_normal_valid_dir = os.path.join(output_path, 'normal', 'valid')
            move_img_to_dir(input_path, valid_normal, target_normal_valid_dir)

    shutil.rmtree(dateset_path)


if __name__ == '__main__':
    build('Class6', 0.8, True)

