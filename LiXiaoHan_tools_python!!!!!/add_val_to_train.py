import json
import os

# 读取原始标注文件
with open("/store/datasets/UA-Detrac/COCO-format/val_b.json", "r") as f:
    data = json.load(f)
with open("/store/datasets/UA-Detrac/COCO-format/train-1-on-5_b.json", "r") as fff:
    data2 = json.load(fff)

# 读取需要筛选的文件名
filter_files = [
    "MVI_39761", "MVI_39781", "MVI_39801", "MVI_39811", "MVI_39821", "MVI_39861",
    "MVI_39931", "MVI_40152", "MVI_40161", "MVI_40162", "MVI_40181", "MVI_40201",
    "MVI_40211", "MVI_40212", "MVI_40213", "MVI_40241", "MVI_40243", "MVI_40244",
    "MVI_40732", "MVI_40751", "MVI_40752", "MVI_40962", "MVI_40991", "MVI_40992",
    "MVI_41063", "MVI_41073", "MVI_63544", "MVI_63552", "MVI_63553", "MVI_63554",
    "MVI_63561", "MVI_63562", "MVI_63563"
]

new_data = {
    "images": [],
    "annotations": [],
    "categories": data["categories"]
}

# 筛选图片
for img in data["images"]:
    img1_path = img["file_name"]
    index = (int(  # img_path中/store/datasets/UA-Detrac/images/MVI_20051/img后的第一个数字（字符的形式）
        img1_path[-9]  # 接上行往后以此类推数字字符
        + img1_path[-8]
        + img1_path[-7]
        + img1_path[-6]
        + img1_path[-5])
             ) % 1000000
    if (index % 5) == 0:
        img["id"] = 76051 + img["id"]
        data2["images"].append(img)

# 筛选标注
for ann in data["annotations"]:
    img_id = ann["image_id"]
    img = next((i for i in data2["images"] if i["id"] == (img_id+76051)), None)
    if img is not None:
        ann["image_id"] = ann["image_id"] +76051
        data2["annotations"].append(ann)

# 输出新的标注文件
with open("/store/datasets/UA-Detrac/COCO-format/train-1-on-5_b_withVal.json", "w") as f:
    json.dump(data2, f)