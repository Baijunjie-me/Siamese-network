# -*- coding:utf-8 -*-
import argparse
import os
import time

from utils import *
import torch
import torch.nn as nn
import logging

" [超参数设置] "
parser = argparse.ArgumentParser(description='PyTorch ISAR Siamese-network')

parser.add_argument('--data_path', type=str, default=r'',
                    help='ISAR data set file path 【】')

parser.add_argument('--n_train_per_class', type=int, default=50,
                    help='number of training sample for per class (default: 50)')

parser.add_argument('--n_val_per_class', type=int, default=50,
                    help='number of val sample for per class (default: 50)')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size  (default: 64)')

parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='train epoch (default: 100)')

parser.add_argument('--best_model_path', type=str, default=r'.\实验结果',
                    help='file path to the model (default: none)')

parser.add_argument('--log_path', type=str, default=r'.\log',
                    help='file path to the model (default: none)')

args = parser.parse_args()

" [日志设置] "
args.time = time.strftime("%Y-%m-%d-%H-%M-%S") # 年月日作为log文件名
args.log_file_path = "{}\{}-log.txt".format(args.log_path, args.time)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

if not os.path.exists(args.log_file_path):
    os.system(r'touch %s' % args.log_file_path)