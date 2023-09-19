from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


import _init_paths

import os
import math

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory
from torch.utils.tensorboard import SummaryWriter
import cv2
import test
import matplotlib.pyplot as pl


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=0,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  # logger.write_model(model)
  print('Starting training...')
  test_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test")
  test_half_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test_self")  #写错了，把half记成了self，只能将错就错下去
  test_recall_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test_recall")
  test_easy_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test_easy")
  test_medium_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test_medium")
  test_hard_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/test_hard")
  train_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/train/loss")
  train_hmloss_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/train/hmloss")
  train_segloss_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/train/segloss")
  val_writer = SummaryWriter(log_dir="runs48.0_alter30.0_attentionMulty_hm/val")
  best = 0.6834
  best_half = 0.7794
  best_easy = 0.8314
  best_medium = 0.7
  best_hard = 0.6
  mAP_LXH = 0
  mAP_LXH_half = 0
  mAP_LXH_easy = 0
  mAP_LXH_medium = 0
  mAP_LXH_hard = 0
  #2023.04.06
  #scheduler_1xh = CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=1, eta_min=5e-6)


  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    train_writer.add_scalar('train_loss', log_dict_train['loss'], epoch)
    train_hmloss_writer.add_scalar('train_hm_loss', log_dict_train['hm_loss'], epoch)
    train_segloss_writer.add_scalar('train_seg_loss', log_dict_train['seg_loss'], epoch)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    opt.model_LXH = model
    mAP_LXH, mAP_LXH_half, mAP_LXH_easy, mAP_LXH_medium, mAP_LXH_hard, recall_mean = test.prefetch_test(opt)
    opt.keep_res = False
    test_writer.add_scalar('test_Accuracy', mAP_LXH, epoch)
    test_half_writer.add_scalar('test_half_Accuracy', mAP_LXH_half, epoch)
    test_recall_writer.add_scalar('test_recall_Accuracy', recall_mean, epoch)
    test_easy_writer.add_scalar('test_easy_Accuracy', mAP_LXH_easy, epoch)
    test_medium_writer.add_scalar('test_medium_Accuracy', mAP_LXH_medium, epoch)
    test_hard_writer.add_scalar('test_hard_Accuracy', mAP_LXH_hard, epoch)
    logger.write('mAP{:3f} | '.format(mAP_LXH))
    if mAP_LXH > best:
      best = mAP_LXH
      save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                 epoch, model)
    if mAP_LXH_half > best_half:
      best_half = mAP_LXH_half
      save_model(os.path.join(opt.save_dir, 'model_best_half.pth'),
                 epoch, model)
    if mAP_LXH_easy > best_easy:
      best_easy = mAP_LXH_easy
      save_model(os.path.join(opt.save_dir, 'model_best_easy.pth'),
                 epoch, model)
    if mAP_LXH_medium > best_medium:
      best_medium = mAP_LXH_medium
      save_model(os.path.join(opt.save_dir, 'model_best_medium.pth'),
                 epoch, model)
    if mAP_LXH_hard > best_hard:
      best_hard = mAP_LXH_hard
      save_model(os.path.join(opt.save_dir, 'model_best_hard.pth'),
                 epoch, model)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      val_writer.add_scalar('val_loss', log_dict_val['loss'], epoch)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  torch.cuda.is_available()
  opt = opts().parse()
  main(opt)