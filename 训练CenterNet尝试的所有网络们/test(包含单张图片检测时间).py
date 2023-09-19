from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

#from lib.external.build.lib.win-amd64-3.7.nms.cp37-win_amd64 import soft_nms
from nms import soft_nms
from lib.opts import opts
from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.datasets.dataset_factory import dataset_factory
from lib.detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in self.opt.test_scales:
      if self.opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  opt.keep_res = True
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Detector = detector_factory[opt.task]
  
  # hughes split = 'val' if not opt.trainval else 'test'
  print("OPT DATASET: ", opt.dataset)
  detector = Detector(opt)
  #开始1
  split = 'test'
  level_LXH = 'results'
  dataset = Dataset(opt, split)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=elapsed, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.save_results(results, opt.save_dir, level_LXH)
  mAP_LXH, mAP_LXH_half, recalll = dataset.run_eval_1(opt.save_dir, level_LXH)
  #结束1
  # 开始4
  split = 'test_easy'
  level_LXH = 'results_easy'
  dataset = Dataset(opt, split)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
      ind, num_iters, total=elapsed, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm=avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.save_results(results, opt.save_dir, level_LXH)
  mAP_LXH_easy = dataset.run_eval(opt.save_dir, level_LXH)
  # 结束4
  # 开始2
  split = 'test_medium'
  level_LXH = 'results_medium'
  dataset = Dataset(opt, split)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
      ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm=avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.save_results(results, opt.save_dir, level_LXH)
  mAP_LXH_medium = dataset.run_eval(opt.save_dir, level_LXH)
  # 结束2
  # 开始3
  split = 'test_hard'
  level_LXH = 'results_hard'
  dataset = Dataset(opt, split)

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process),
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
      ind, num_iters, total=elapsed, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm=avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.save_results(results, opt.save_dir, level_LXH)
  mAP_LXH_hard = dataset.run_eval(opt.save_dir, level_LXH)
  # 结束3
  return mAP_LXH, mAP_LXH_half, mAP_LXH_easy, mAP_LXH_medium, mAP_LXH_hard

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  # hughes split = 'val' if not opt.trainval else 'test'
  split = 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)