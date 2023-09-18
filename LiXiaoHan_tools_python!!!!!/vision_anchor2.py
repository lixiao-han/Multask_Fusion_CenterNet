import torchvision
from PIL import ImageDraw
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取JSON文件
with open("/store/datasets/UA-Detrac/COCO-format/test-1-on-30_b_medium.json", 'r') as f:
    data = json.load(f)

with open("/store/datasets/UA-Detrac/exp/ctdet/coco_dla_1x2.0_43.1_train-1-on-5_b_withVal/results_medium.json", 'r') as f2:
    results = json.load(f2)

i = 0
for im in data["images"]:
    imagepth = im["file_name"]
    image = Image.open(imagepth)
    image_handler = ImageDraw.ImageDraw(image)
    for result in results:
        if result["image_id"] == im["id"] and result["score"]>=0.5:
            bbox = result["bbox"]
            # 绘制红色矩形
            image_handler.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])),
                                    outline="red", width=5)
            # 在矩形上方绘制文本标签
            label = "vehicle"
            image_handler.text((bbox[0] + 5, bbox[1] - 15), label, fill="red")
    image.save(imagepth)
