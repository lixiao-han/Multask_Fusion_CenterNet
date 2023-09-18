import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
import os

# 读取原始数据集的标注信息
with open("/store/datasets/UA-Detrac/COCO-format/annotations_aug2.json", "r") as fff:
    dataset = json.load(fff)

# 设置数据增强参数
params = [
    # {"type": "random_crop", "width": 960, "height": 540},
    {"type": "horizontal_flip"},
    {"type": "random_rotation", "angle_range": (-10, 10)},
    # {"type": "brightness_adj", "brightness_range": (0.5, 1.5)},
    {"type": "contrast_adj", "contrast_range": (0.5, 1.5)}
]

# 定义数据增强函数
def augment(img, img1, annotations):
    # 图像增强
    for param in params:
        if param["type"] == "horizontal_flip":
            img = cv2.flip(img, 1)
            img1 = cv2.flip(img1, 1)
            # 更新目标框坐标
            for ann in annotations:
                ann["bbox"][0] = img.shape[1] - ann["bbox"][0] - ann["bbox"][2]
            for i in range(4):
                ann["bbox"][i] = round(ann["bbox"][i], 2)

        elif param["type"] == "random_rotation":
            angle = np.random.uniform(param["angle_range"][0], param["angle_range"][1])
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
            # 更新目标框坐标
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                cx, cy = x + w/2, y + h/2
                rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
                xy = np.array([cx, cy])
                xy -= np.array([img.shape[1]/2, img.shape[0]/2])
                xy = np.dot(rotation_matrix, xy)
                xy += np.array([img.shape[1]/2, img.shape[0]/2])
                ann["bbox"][0] = xy[0] - ann["bbox"][2]/2
                ann["bbox"][1] = xy[1] - ann["bbox"][3]/2
                for i in range(4):
                    ann["bbox"][i] = round(ann["bbox"][i], 2)

        elif param["type"] == "brightness_adj":
            img = cv2.convertScaleAbs(img, alpha=param["brightness_range"][0], beta=np.random.uniform(param["brightness_range"][0], param["brightness_range"][1]) * 255)
            img1 = cv2.convertScaleAbs(img1, alpha=param["brightness_range"][0], beta=np.random.uniform(param["brightness_range"][0], param["brightness_range"][1]) * 255)

        elif param["type"] == "contrast_adj":
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(param["contrast_range"][0], param["contrast_range"][1]), beta=0)

    return img, img1, annotations

dataset2 = {
"images": [],
"annotations": [],
"categories": []
}

dataset3 = {
"images": [],
"annotations": [],
"categories": []
}

ann_num = 25707
# 打开输出文件
with open("/store/datasets/UA-Detrac/COCO-format/annotations_aug1.json", "w") as f:
    # 遍历原始数据集
    for i, img_ann in enumerate(dataset["images"]):
        # 读取原始图片
        img = cv2.imread(img_ann["file_name"])
        img1 = cv2.imread(os.path.join('/store/datasets/UA-Detrac/pyflow-bgsubs',
                              os.path.dirname(img_ann["file_name"]).split('/')[-1],
                              os.path.basename(img_ann["file_name"]).replace('jpg', 'png')))
        the_id = img_ann["id"]
        ann_aug1 = []
        for ann in dataset["annotations"]:
            if ann["image_id"] == the_id:
                ann_aug1.append(ann)
        img_aug, img1_aug, ann_aug = augment(np.copy(img), np.copy(img1), ann_aug1)
        index = str(i+1).zfill(5)
        # 输出新图片
        cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/images/aug_images", index + ".jpg"),  img_aug)
        cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/pyflow-bgsubs/aug_images", index + ".png"), img1_aug)
        iimg_aug = {"file_name": os.path.join("/store/datasets/UA-Detrac/images/aug_images", index + ".jpg"),
                    "id": i + 4814,
                    "calib": ""}
        dataset3["images"].append(iimg_aug)
        # 更新标注信息中的文件名和ID
        for ann in ann_aug:
            ann_num += 1
            ann["image_id"] = iimg_aug["id"]
            ann["id"] = ann_num
            dataset2["annotations"].append(ann)
    with open("/store/datasets/UA-Detrac/COCO-format/annotations_aug3.json", "w") as f3:
        json.dump(dataset3, f3)
    json.dump(dataset2, f)
    f.write("\n")

