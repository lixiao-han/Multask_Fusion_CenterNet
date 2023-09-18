
from PIL import ImageDraw
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取JSON文件
with open("/store/datasets/UA-Detrac/COCO-format/MVI_40161.json", 'r') as f:
    data = json.load(f)

i = 0
for im in data["images"]:
    imagepth = im["file_name"]
    image = Image.open(imagepth)
    image_handler = ImageDraw.ImageDraw(image)
    for result in data["annotations"]:
        if result["image_id"] == im["id"]:
            bbox = result["bbox"]
            # 绘制红色矩形
            image_handler.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])),
                                    outline="red", width=5)
            # 在矩形上方绘制文本标签
            label = "vehicle"
            image_handler.text((bbox[0] + 5, bbox[1] - 15), label, fill="red")
    image.save(imagepth)