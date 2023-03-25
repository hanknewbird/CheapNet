from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# coco格式的json文件，原始标注数据
anno_file = '../../../Data/DAGM/Class6/Annotations/valid.json'

data_classes = [6]
groups = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# thresholds = [0.5]
# [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# groups = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for data_class in data_classes:
    for threshold in thresholds:
        for group in groups:

            predict_file = f'../../../Data/DAGM/Class{data_class}/Annotations/threshold_{threshold}/predict_G{group}.json'

            coco_gt = COCO(anno_file)

            # 用GT框作为预测框进行计算，目的是得到detection_res
            with open(predict_file, 'r') as f:  # 打开json
                json_file = json.load(f)  # 解析json
            annotations = json_file['annotations']  # 得到json中的annotations分支
            detection_res = []

            for anno in annotations:
                detection_res.append({
                    'score': 1.,
                    'category_id': anno['category_id'],
                    'bbox': anno['bbox'],
                    'image_id': anno['image_id']
                })

            print(f"↓↓↓↓↓↓↓↓↓↓↓ Threshold:{threshold} Group:{group} ↓↓↓↓↓↓↓↓↓↓↓")

            # image要对应
            coco_dt = coco_gt.loadRes(detection_res)

            cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')  # 初始化CocoEval对象
            cocoEval.evaluate()  # 按图像评估运行
            cocoEval.accumulate()  # 为图像结果累积
            cocoEval.summarize()  # 显示结果的汇总指标
            print(cocoEval.stats)

            print(f"↑↑↑↑↑↑↑↑↑↑↑ Threshold:{threshold} Group:{group} ↑↑↑↑↑↑↑↑↑↑↑")
