import os
import shutil
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def get_file_names(path = "/store/datasets/UA-Detrac/COCO-format/test-1-on-30_b_medium.json"):
    images = set()   #该语句来保证集合images中不能包含重复的元素
    with open(path, 'r') as f:
        data = json.load(f)
    for img in data["images"]:
        images.add(img["file_name"])
    return images

filenames = get_file_names("/store/datasets/UA-Detrac/COCO-format/test-1-on-30_b.json")
for filename in filenames:
    dst_file = os.path.join('/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_30all', os.path.dirname(filename).split('/')[-1], os.path.basename(filename))
    dst_file1 = os.path.join('/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_30all',
                            os.path.dirname(filename).split('/')[-1])
    os.makedirs(dst_file1, exist_ok=True)
    shutil.copy2(filename, dst_file)

