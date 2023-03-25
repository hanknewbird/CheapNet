#!/usr/bin/env python3
import datetime
import json
import os
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = os.path.join("..", "..", "..", "Data", "DAGM", "Class10")  # 路径
IMAGE_DIR = os.path.join(ROOT_DIR, "abnormal")  # img
LABEL_DIR = os.path.join(ROOT_DIR, "abnormal_label")  # label

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


def main():
    # 通过label寻找对应image
    for root, _, files in os.walk(LABEL_DIR):
        dataset = os.path.basename(root)

        if dataset == "train" or dataset == "valid":
            INFO['description'] = f"DAGM abnormal COCO {dataset} dataset"
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

            for file in files:
                file_path = os.path.join(root, file)
                # print(dataset, file_path)
                image = Image.open(file_path)  # 打开图像
                image_info = pycococreatortools.create_image_info(image_id, os.path.basename(file_path).replace("_label", ""), image.size)  # 创建images
                coco_output["images"].append(image_info)  # 追入coco_output

                category_info = {'id': 1, 'is_crowd': 0}  # 只有1个类
                binary_mask = np.asarray(Image.open(file_path).convert('1')).astype(np.uint8)  # 以二值图像读取
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask, image.size, tolerance=2)  # 创建annotation
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id += 1
                image_id += 1

            create_dir(f"{ROOT_DIR}/Annotations")
            with open(f'{ROOT_DIR}/Annotations/{dataset}.json', 'w') as output_json_file:
                json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
