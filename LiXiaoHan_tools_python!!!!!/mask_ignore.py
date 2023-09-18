import torchvision
from PIL import ImageDraw
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# 导入所需库
import os, random, shutil
import cv2
from math import *
import numpy as np
import os.path as osp
import PIL

with open("/store/datasets/UA-Detrac/COCO-format/test-1-on-30_b_medium.json", 'r') as f:
    data = json.load(f)
source_file = "/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_30all"
result_imgs_dir = "/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_mask_ignore"
mask_corodinary = {"MVI_39031": [[0, 0, 0, 0]], "MVI_39051": [[912, 238, 952, 287]], "MVI_39211": [[622, 5, 953, 54]],
                   "MVI_39271": [[884, 153, 954, 198], [725, 362, 956, 539]], "MVI_39311": [[28, 242, 368, 275], [705, 144, 950, 244], [811, 316, 955, 418]],
                   "MVI_39361": [[89, 28, 882, 179]], "MVI_39371": [[13, 306, 323, 357], [832, 227, 954, 283]], "MVI_39401": [[134, 58, 725, 152]],
                   "MVI_39501": [[762, 171, 954, 339], [672, 170, 760, 268], [598, 131, 662, 197], [106, 134, 270, 357]], "MVI_39511": [[762, 171, 954, 339], [672, 170, 760, 268], [598, 131, 662, 197], [106, 134, 270, 357]],
                   "MVI_40701": [[7, 8, 124, 155], [366, 4, 934, 72]], "MVI_40711": [[273, 4, 655, 108]], "MVI_40712": [[258, 3, 599, 104]], "MVI_40714": [[369, 24, 673, 100]],
                   "MVI_40742": [[146, 65, 692, 130], [465, 123, 949, 173]], "MVI_40743": [[2, 74, 214, 172]], "MVI_40761": [[0, 0, 0, 0]], "MVI_40762": [[0, 0, 0, 0]],  "MVI_40763": [[0, 0, 0, 0]],
                   "MVI_40771": [[336, 5, 687, 45], [694, 7, 955, 104]], "MVI_40772": [[0, 0, 0, 0]], "MVI_40773": [[0, 0, 0, 0]], "MVI_40774": [[746, 3, 955, 88]], "MVI_40775": [[751, 3, 959, 144]],
                   "MVI_40792": [[779, 7, 957, 72]], "MVI_40793": [[606, 10, 951, 86]], "MVI_40851": [[63, 34, 940, 102]], "MVI_40852": [[231, 17, 911, 102]], "MVI_40853": [[231, 17, 911, 102]],
                   "MVI_40854": [[231, 17, 911, 102]], "MVI_40855": [[2, 5, 950, 100]], "MVI_40863": [[2, 12, 75, 91], [71, 6, 928, 69]], "MVI_40864": [[2, 12, 75, 91], [71, 6, 928, 69]],
                   "MVI_40891": [[5, 9, 953, 64]], "MVI_40892": [[318, 6, 829, 58]], "MVI_40901": [[3, 6, 310, 74]], "MVI_40902": [[8, 86, 187, 152], [171, 5, 375, 29], [5, 300, 955, 520]],
                   "MVI_40903": [[3, 6, 310, 74]], "MVI_40904": [[4, 7, 290, 81], [7, 95, 230, 238], [5, 433, 955, 538]], "MVI_40905": [[4, 7, 290, 81], [7, 95, 230, 238]]
                   }

folder_path = "/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_30all"  # 替换为你要遍历的文件夹路径
image_dirs = [f for f in os.listdir(folder_path)]
for image_file in image_dirs:
    JueDuiDirpth = os.path.join("/store/datasets/UA-Detrac/Insight-MVT_Annotation_Test_30all", image_file)
    images_pth = [f for f in os.listdir(JueDuiDirpth) if f.endswith((".jpg", ".png", ".jpeg"))]
    cor_list = mask_corodinary.get(image_file)
    for image_pth in images_pth:
        JueDuiimages_pth = os.path.join(JueDuiDirpth, image_pth)
        M_Img = cv2.imread(JueDuiimages_pth, 3)
        for cor in cor_list:
            M_Img[cor[1]:cor[3], cor[0]:cor[2], 0:3] = 0
        result_imgs_th = os.path.join(result_imgs_dir, image_file, image_pth)
        result_imgs_th1 = os.path.join(result_imgs_dir, image_file)
        os.makedirs(result_imgs_th1, exist_ok=True)
        cv2.imwrite(result_imgs_th, M_Img)



