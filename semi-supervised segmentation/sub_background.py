import cv2
import numpy as np
import os
import csv
from PIL import Image

# 创建目标文件夹
if not os.path.exists('/store/datasets/UA-Detrac/foreground2'):
    os.makedirs('/store/datasets/UA-Detrac/foreground2')
'''
def get_pairs_from_list(list_names=['/store/datasets/UA-Detrac/val-tf-all.csv']):

    images = set()   #该语句来保证集合images中不能包含重复的元素
    for list_name in list_names:
        with open(list_name) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                img1_path = row[0]  #比如 对于detrac数据集的train-tf-all.csv文件是  img1_path = /store/datasets/UA-Detrac/images/MVI_20051/img00001.jpg
                images.add(img1_path)
    return images
'''
# 打开视频文件
#cap = cv2.VideoCapture('your_video.mp4')
uaimages = []
uaimages = os.listdir('/store/datasets/UA-Detrac/images1')
for mvi in uaimages:
    mviimages = []
    mviimages = os.listdir(os.path.join('/store/datasets/UA-Detrac/images', mvi))
    mvi_len = len(mviimages)
    '''
    if mvi_len < 600:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
    elif mvi_len >= 600 and mvi_len < 700:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=30, detectShadows=False)
    elif mvi_len >= 700 and mvi_len < 800:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=30, detectShadows=False)
    elif mvi_len >= 800 and mvi_len < 900:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=30, detectShadows=False)
    elif mvi_len >= 900 and mvi_len < 1000:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=900, varThreshold=30, detectShadows=False)
    elif mvi_len >=1000:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=30, detectShadows=False)
    '''
    fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=10, detectShadows=False)
    images = set()
    for list_name in mviimages:
        img1_path = os.path.join('/store/datasets/UA-Detrac/images', mvi, list_name)
        images.add(img1_path)
    images = sorted(images)
    for im in images:
        frame = cv2.imread(im)

        # 背景subtraction
        fgmask = fgbg.apply(frame)

        # 进行二值化和形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # 对前景进行处理
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        fgmask_bin = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        fgmask_bin[thresh == 255] = 255

        taget_doc = os.path.join('/store/datasets/UA-Detrac/foreground2', mvi,
                                 os.path.basename(im).replace('jpg', 'png'))
        if not os.path.exists(os.path.dirname(taget_doc)):
            os.mkdir(os.path.dirname(taget_doc))
        # 生成前景图
        cv2.imwrite(taget_doc, fgmask_bin)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
'''
# 创建背景subtraction对象
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
images = get_pairs_from_list()
images = sorted(images)

for im in images:
    frame = cv2.imread(im)

    # 背景subtraction
    fgmask = fgbg.apply(frame)

    # 进行二值化和形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # 对前景进行处理
    _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    fgmask_bin = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    fgmask_bin[thresh == 255] = 255



    taget_doc = os.path.join('/store/datasets/UA-Detrac/foreground', os.path.dirname(im).split('/')[-1], os.path.basename(im).replace('jpg', 'png'))
    if not os.path.exists(os.path.dirname(taget_doc)):
        os.mkdir(os.path.dirname(taget_doc))
    # 生成前景图
    cv2.imwrite(taget_doc, fgmask_bin)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
import numpy as np
import cv2

# 创建 Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# 打开视频文件
cap = cv2.VideoCapture('/store/datasets/UA-Detrac/images/output.avi')

while True:
    ret, frame = cap.read() # 读取每一帧图像
    if ret:

        # 应用 background subtraction
        fgmask = fgbg.apply(frame)

        # 对二值图像进行形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 二值化分割图
        _, thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)
        fgmask_bin = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        fgmask_bin[thresh == 255] = 255

        # 显示二值化分割图
        cv2.imshow('fgmask_bin', fgmask_bin)

    else:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 清理
cap.release()

'''