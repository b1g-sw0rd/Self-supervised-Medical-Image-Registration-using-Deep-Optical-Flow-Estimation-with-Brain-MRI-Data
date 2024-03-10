import argparse
from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
import os
from torchvision.io import read_image
import torch
import torch.nn as nn
from models import opticalFlowReg
from utils import averageMeter
from loss import OFEloss
import time
from dataset import brain_dataset

# 调用gpu计算如果可用，否则cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 打印间隔
PRINT_INTERVAL = 8


def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = averageMeter()
    avg_batch_time = averageMeter()
    avg_smooth_loss = averageMeter()
    avg_photo_loss = averageMeter()
    avg_corr_loss = averageMeter()
    i = 0
    tic = time.time()
    for imgs in data:  # imgs 包含前1通道fixed 后1 moving

        imgs = imgs.to(device)
        imgs = imgs.float()
        with torch.set_grad_enabled(optimizer is not None):  # 控制梯度计算的开启关闭
            pred_flows, wraped_imgs = model(imgs)
            fixed_img = torch.unsqueeze(imgs[:, 0, :, :], dim=1)  # 恢复通道维度
            photo_loss, corr_loss, smooth_loss, loss = criterion(pred_flows, wraped_imgs, fixed_img)
            # imgs 包含前1通道fixed 后1 moving
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time = time.time() - tic
        tic = time.time()
        avg_photo_loss.update(photo_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_corr_loss.update(corr_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0 or i+1 == len(data):
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
                  'corr_loss {corr.val:5.4f} ({corr.avg:5.4f})\t'
                  'photo_loss {photo.val:5.4f} ({photo.avg:5.4f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i+1, len(data), batch_time=avg_batch_time, loss=avg_loss,
                smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))
        i += 1

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg corr_loss {corr.avg:5.4f} \t'
          'Avg photo_loss {photo.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))

    return avg_photo_loss.avg, avg_corr_loss.avg, avg_smooth_loss.avg, avg_loss.avg


if __name__ == '__main__':
    # 参数读取
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r"D:\OASIS", type=str, metavar='DIR',
                        help='path to dataset')  # 修改
    parser.add_argument('--model', default='flowNetS', type=str, help='the supervised model to be trained with ('
                                                                     'flowNetS, lightflownet, pwc_net)')  # 修改
    parser.add_argument('--steps', default=480, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--batch_size', default=10, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', default=1.6e-5, type=float, metavar='LR', help='learning rate')

    args = parser.parse_args()

    #  模型加载
    OFEmodel = opticalFlowReg(conv_predictor=args.model)
    OFEmodel.to(device)

    path = os.path.join("Unsupervised", type(OFEmodel.predictor).__name__)
    loss_fnc = OFEloss
    optim = torch.optim.Adam(OFEmodel.parameters(), args.lr, betas=(0.9, 0.999))
    data_dir = r"D:\OASIS"
    train_generator = data.DataLoader(brain_dataset("D:/test/imgA", "D:/test/imgB"), batch_size=args.batch_size, shuffle=False)
    train_length = len(train_generator)
    epochs = args.steps // train_length
    starting_epoch = 0
    for e in range(starting_epoch, epochs):
        print("=================\n EPOCH " + str(e + 1) + "/" + str(epochs) + " \n=================\n")
        print("")
        photo_loss, corr_loss, smooth_loss, total_loss = epoch(OFEmodel, train_generator, loss_fnc, optim)
        with torch.no_grad():
            photo_loss, corr_loss, smooth_loss, total_loss = epoch(OFEmodel, train_generator, loss_fnc, optim)
