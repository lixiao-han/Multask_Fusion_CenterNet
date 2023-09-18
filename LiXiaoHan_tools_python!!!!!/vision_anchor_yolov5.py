import torchvision
from PIL import ImageDraw
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

with open("/PyCharm_project/yolov5-5.0_2.0/runs3.0.1.2_hard/train/exp/_predictions.json", 'r') as f2:
    results = json.load(f2)

folder_path = "/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test(YOLOV5)"  # 替换为你要遍历的文件夹路径
image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]

for image_file in image_files:
    file_id = image_file.split('.')[-2]
    imagepth = os.path.join("/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test(YOLOV5)", image_file)
    image = Image.open(imagepth)
    image_handler = ImageDraw.ImageDraw(image)
    for result in results:
        if result["image_id"] == file_id and result["score"]>=0.5:
            bbox = result["bbox"]
            # 绘制红色矩形
            image_handler.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])),
                                    outline="red", width=5)
            # 在矩形上方绘制文本标签
            label = "vehicle"
            image_handler.text((bbox[0] + 5, bbox[1] - 15), label, fill="red")
    image.save(imagepth)
