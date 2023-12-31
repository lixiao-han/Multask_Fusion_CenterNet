from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from ...utils.image import flip, color_aug
from ...utils.image import get_affine_transform, affine_transform
from ...utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from ...utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    the_lens = len(self.images)
    coco_loadImags_ = self.coco.loadImgs(ids=[img_id])
    file_name = coco_loadImags_[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    if 'uav' in self.opt.dataset:
      seg_path = os.path.join('/store/datasets/UAV/bgsubs',
                              os.path.dirname(file_name).split('/')[-1],
                              os.path.basename(file_name).replace('jpg', 'png'))
    else:
      seg_path = os.path.join('/store/datasets/UA-Detrac/pyflow-bgsubs',
                              os.path.dirname(file_name).split('/')[-1],
                              os.path.basename(file_name).replace('jpg', 'png'))
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    seg_img = cv2.imread(seg_path, 0)  # hughes
    img = cv2.imread(img_path)
    # print("IMG_SHAPE: ", img.shape, " MEAN: ", np.mean(img))
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", os.path.basename(file_name)), img)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        seg_img = seg_img[:, ::-1]
        # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), img)
        # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), seg_img)
        c[0] =  width - c[0] - 1


    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    # print('TRANS INPUT SHAPE: ', trans_input.shape)
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    seg_inp = cv2.warpAffine(seg_img, trans_input,
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)

    seg_inp = (seg_inp.astype(np.float32) / 255.) # hughes
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    #这两行代码是数据增强的操作，如果数据集类型为训练集且没有设置不进行颜色增强（no_color_aug），那么会执行color_aug函数来对图片进行颜色增强操作。
    #其中，self._data_rng是数据随机数生成器，inp是输入的图片数据，并且self._eig_val和self._eig_vec是预处理的颜色空间的特征值和特征向量，用于进行归一化操作。这些参数的预处理可以使得颜色增强操作的效果更加平稳自然。
    #在color_aug函数中，会对图片进行一系列的颜色变换，包括亮度、对比度、色调、饱和度等方面的变换，从而增加数据集的多样性，提高模型的泛化能力。


    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), inp)
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), seg_inp)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    # print('MEAN: ', np.average(seg_inp))

    output_h = input_h // self.opt.down_ratio    #下采样  比如CenterNet模型得到的输出是128*128
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'seg': np.expand_dims(seg_inp, 0)}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta

    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), (inp.transpose(1, 2, 0)* 255).astype(np.uint8))
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), (seg_inp * 255).astype(np.uint8))

    return ret
