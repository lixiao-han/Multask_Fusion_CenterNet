import os
import numpy as np
from lxml import etree
from PIL import Image
import xml.etree.ElementTree as ET
import os





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

# Directory path to loop through
directory = r'E:\store\datasets\UA-Detrac\DETRAC-Train-Annotations-XML1'

# Initialize an empty list to store the file names
files = []

# Loop through each file in the directory
for file in os.listdir(directory):
    # Check if the path is a file and ends with .txt, then add it to the list
    if os.path.isfile(os.path.join(directory, file)) and file.endswith('.xml'):
        files.append(os.path.join(directory, file))


for file in files:
    lxxh = 0
    file_name = os.path.basename(file[:-4])
    # Load the XML file
    tree = ET.parse(file)
    root = tree.getroot()
    pyflow_bgsub_name = os.path.join('/store/datasets/UA-Detrac/pyflow-bgsubs', file_name)
    num_total = 0
    print(file_name + ':')
    # Loop through each frame in the sequence
    for frame in root.findall('.//frame'):
        num = int(frame.get('num'))
        density = int(frame.get('density'))
        if density < 5:
            total_area = 0
            for target in frame.findall('.//target'):
                target_id = target.get('id')
                box = target.find('box')
                left, top, width, height = [int(float(box.get(attr))) for attr in ['left', 'top', 'width', 'height']]
                area = width*height
                total_area += area
            if total_area < 2500:
                print(str(num).zfill(5) + ',')
        if num != 1:
            if (num-lxxh) == 2:
                print(str(lxxh+1).zfill(5) + ',')
            if (num-lxxh) > 2:
                print(str(lxxh+1).zfill(5) + '----' + str(num-1).zfill(5) + ',')
        lxxh = num

