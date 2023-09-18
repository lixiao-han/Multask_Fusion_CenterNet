import os
import numpy as np
from lxml import etree
from PIL import Image
import xml.etree.ElementTree as ET
import os





# Load the XML file
tree = ET.parse('filename.xml')
root = tree.getroot()


def binary(image_array, threshold):
    """
    对给定的图像数组进行二值化。

    Parameters:
    ----------
    image_array : numpy array
        输入的图像数组。
    threshold : float
        二值化的阈值。

    Returns：
    -------
    numpy array
        二值化后的图像数组。
    """
    binary_image = np.zeros_like(image_array)
    binary_image[image_array > threshold] = 1
    return binary_image


def calculate_iou(box1, box2):
    """
    计算两个矩形框之间的 IoU

    Parameters:
    ----------
    box1 : tuple
        第一个矩形框，格式为 (xmin, ymin, xmax, ymax)
    box2 : tuple
        第二个矩形框，格式为 (xmin, ymin, xmax, ymax)

    Returns:
    -------
    float
        两个矩形框之间的 IoU 值。
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB -yA + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

# Directory path to loop through
directory = r'E:\store\datasets\UA-Detrac\DETRAC-Train-Annotations-XML'

# Initialize an empty list to store the file names
files = []

# Loop through each file in the directory
for file in os.listdir(directory):
    # Check if the path is a file and ends with .txt, then add it to the list
    if os.path.isfile(os.path.join(directory, file)) and file.endswith('.txt'):
        files.append(file)


for file in files:
    recall_total = 0
    file_name = file[:-4]
    # Load the XML file
    tree = ET.parse(file)
    root = tree.getroot()
    pyflow_bgsub_name = os.path.join('/store/datasets/UA-Detrac/pyflow-bgsubs', file_name)
    pys = []
    for file_img in os.listdir(pyflow_bgsub_name):
        # Check if the path is a file and ends with .txt, then add it to the list
        if os.path.isfile(os.path.join(pyflow_bgsub_name, file_img)) and file_img.endswith('.png'):
            pys.append(file_img)
    # Loop through each frame in the sequence
    for frame in root.findall('.//frame'):
        num = frame.get('num')
        boxes = []
        if num % 10 == 0:
            # Loop through each target in the frame
            for target in frame.findall('.//target'):
                target_id = target.get('id')
                box = target.find('box')
                left, top, width, height = [int(box.get(attr)) for attr in ['left', 'top', 'width', 'height']]
                right = left + width
                down = top + height
                boxes.append((left, top, right, down))
            if len(boxes) == 0:
                continue
            else:
                recall = 0
                total = len(boxes)
                binary_pred_array = pys[num-1]
                for box_true in boxes:
                    if np.sum(binary_pred_array[box_true[1]:box_true[3], box_true[0]:box_true[2]]) > 0:
                        recall += 1
                recall_rate = recall/total
                recall_total += recall_rate
    print(file_name + '的总召回率为：' + recall_total/(len(pys)/10))






def get_boxes_from_annotation(annotation_path):
    """
    从 XML 格式的注释文件中提取矩形框。

    Parameters:
    ----------
    annotation_path : str
        XML 格式的注释文件的路径。

    Returns:
    -------
    list
        包含所有矩形框的列表。每个矩形框表示为 (xmin, ymin, xmax, ymax) 格式的元组。
    """
    with open(annotation_path) as f:
        xml_content = f.read()
    xml_root = etree.fromstring(xml_content)
    boxes = []
    for object_node in xml_root.xpath('object'):
        box_node = object_node.xpath('bndbox')[0]
        xmin = int(box_node.xpath('xmin')[0].text)
        ymin = int(box_node.xpath('ymin')[0].text)
        xmax = int(box_node.xpath('xmax')[0].text)
        ymax = int(box_node.xpath('ymax')[0].text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def calculate_recall(pred_dir, anno_dir, threshold):
    """
    计算给定目录中预测文件和真实标注文件之间的召回率。

    Parameters:
    ----------
    pred_dir : str
        包含所有二值化的分割图像的目录路径。
    anno_dir : str
        包含所有 XML 格式的 PASCAL VOC 标注文件的目录路径。
    threshold : float
        二值化的阈值。

    Returns:
    -------
    tuple
        召回率值和总数。
    """
    fns_pred = os.listdir(pred_dir)
    recall = 0
    total = 0
    for fn_pred in fns_pred:
        fn_anno = fn_pred[:-3] + 'xml'
        pred_path = os.path.join(pred_dir, fn_pred)
        anno_path = os.path.join(anno_dir, fn_anno)
        boxes_true = get_boxes_from_annotation(anno_path)
        if len(boxes_true) == 0:
            continue
        with Image.open(pred_path).convert('L') as pred_image:
            pred_array = np.array(pred_image, dtype=np.float32)
            binary_pred_array = binary(pred_array, threshold)
            for box_true in boxes_true:
                if np.sum(binary_pred_array[box_true[1]:box_true[3], box_true[0]:box_true[2]]) > 0:
                    recall += 1
                    break
        total += 1
    return recall / total, total