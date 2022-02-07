# -*- coding:utf-8 -*-
import argparse
import os
import time
from train import *
import torch
import torch.nn as nn
import logging
from data_loader import dataLoader
# 用来控制dataload类型
BY_N_PER_CLASS = 0
BY_PERCENT_PER_CLASS = 1

" [超参数设置] "
parser = argparse.ArgumentParser(description='PyTorch ISAR Siamese-network')

parser.add_argument('--data_path', type=str, default=r'MSTAR-10/data/',
                    help='ISAR data set file path')

parser.add_argument('--n_train_per_class', type=int, default=50,
                    help='number of train sample for per class (default: 50)')

parser.add_argument('--n_val_per_class', type=int, default=50,
                    help='number of val sample for per class (default: 50)')

parser.add_argument('--train_class_percent', type=int, default=0.1,
                    help='percent of train sample for per class (default: 50)')

parser.add_argument('--val_class_percent', type=int, default=0.1,
                    help='percent of val sample for per class (default: 50)')

parser.add_argument('--load_data_type', type=int, default=BY_N_PER_CLASS,
                    help='load dat type '
                         'BY_N_PER_CLASS --> 按照个数 / '
                         'BY_PERCENT_PER_CLASS --> 按半分比 (default: 50)')

parser.add_argument('--lr', type=float, default=0.003, metavar='N',
                    help='leraning rate (default: 0.003)')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size  (default: 64)')

parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='train epoch (default: 100)')

parser.add_argument('--best_model_path', type=str, default=r'.\实验结果',
                    help='file path to the model (default: none)')

parser.add_argument('--cuda', type=bool, default=False,
                    help='cuda is available or not (default: Flase)')

parser.add_argument('--log_path', type=str, default=r'.\log',
                    help='file path to the model (default: none)')

args = parser.parse_args()

if __name__ == "__main__":
    " [日志设置] "
    args.time = time.strftime("%Y-%m-%d-%H-%M-%S") # 年月日作为log文件名
    args.log_file_path = "{}\{}-log.txt".format(args.log_path, args.time)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.log_file_path):
        os.system(r'touch %s' % args.log_file_path)

    args.cuda = torch.cuda.is_available()

    dataLoader = dataLoader(args)
    args.train_dataloader, args.val_dataloader, args.test_dataloader = dataLoader.load_mstar_data()
    # args.train_dataloader, args.val_dataloader, args.test_dataloader = dataLoader.MNIST_data_loader(mean = 0.1307, std = 0.3081)

    train(args)



