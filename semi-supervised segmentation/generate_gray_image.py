import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
import os
import matplotlib.pyplot as plt


with open("/store/datasets/UA-Detrac/COCO-format/train-1-on-10_b.json", "r") as fff:
    dataset = json.load(fff)

for i, img_ann in enumerate(dataset["images"]):
    file_name = img_ann["file_name"]
    # 读取原始图片
    optical_image_path = os.path.join('/store/datasets/UA-Detrac/pyflow',
                              os.path.dirname(file_name).split('/')[-1],
                              os.path.basename(file_name))
    gray_img = cv2.imread(optical_image_path, 0)

    # 对前景掩码进行形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #开运算：使用cv2.morphologyEx()函数的第二个参数为cv2.MORPH_OPEN时，将通过传入的结构元素(即kernel)对图像进行腐蚀操作，
    #然后再通过该结构元素对结果进行膨胀操作。这种运算可以用于去除小的噪声点和填补小的空洞。
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)

    #闭运算：使用cv2.morphologyEx()函数的第二个参数为cv2.MORPH_CLOSE时，将通过传入的结构元素(即kernel)对图像进行膨胀操作，
    #然后再通过该结构元素对结果进行腐蚀操作。这种运算可以用于填补小的断裂和连接相邻的物体。
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

    output_path = os.path.join('/store/datasets/UA-Detrac/pyflow_gray',
                              os.path.dirname(file_name).split('/')[-1],
                              os.path.basename(file_name))
    if not os.path.exists(output_path):

        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))

        cv2.imwrite(output_path, gray_img)
