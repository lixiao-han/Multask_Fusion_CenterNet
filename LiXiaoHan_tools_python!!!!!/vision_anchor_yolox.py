import torchvision
from PIL import ImageDraw
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取JSON文件
with open("/PyCharm_project/YOLOX1/datasets/COCO/annotations/instances_test2017.json", 'r') as f:
    data = json.load(f)

with open("/PyCharm_project/YOLOX1/tools/yolox_testdev_2017.json", 'r') as f2:
    results = json.load(f2)

i = 0
for im in data["images"]:
    imagepth = os.path.join("/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test(YOLOX)", im["file_name"])
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
