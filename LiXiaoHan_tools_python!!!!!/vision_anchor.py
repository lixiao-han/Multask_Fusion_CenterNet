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
for result in results:
    #i+=1
    imgpth = next((i for i in data["images"] if i["id"] == result["image_id"]), None)["file_name"]
    image = Image.open(imgpth)
    bbox = result["bbox"]
    # 创建矩形对象并添加到图像上
    #fig, ax = plt.subplots(1)
    #ax.imshow(image)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # 去除坐标轴刻度
    ax.axis('off')
    plt.savefig(imgpth, bbox_inches='tight')
    # 关闭图像窗口
    plt.close()

