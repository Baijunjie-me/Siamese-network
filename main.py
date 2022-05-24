# -*- coding:utf-8 -*-
import argparse
import os
import time
from train import *
from test import *
import torch
from Logger import *
from data_loader import dataLoader
# 用来控制dataload类型
BY_N_PER_CLASS = 0
BY_PERCENT_PER_CLASS = 1


" [超参数设置] "
parser = argparse.ArgumentParser(description='PyTorch ISAR Siamese-network')

parser.add_argument('--need_train', type=bool, default=True,
                    help='default need train , save model ')

parser.add_argument('--data_path', type=str, default=r'E:\\研究生毕设\\舰船实测数据\\ship_jpg\\',
                    help='ISAR data set file path')

parser.add_argument('--model_path', type=str, default='E:\\研究生毕设\\isar程序\\Siamese-network\\Siamese-network\\net.pt',
                    help='trained model')

parser.add_argument('--n_train_per_class', type=int, default=50,
                    help='number of train sample for per class (default: 50)')

parser.add_argument('--n_val_per_class', type=int, default=50,
                    help='number of val sample for per class (default: 50)')

parser.add_argument('--train_class_percent', type=float, default=0.3,
                    help='percent of train sample for per class (default: 50)')

parser.add_argument('--val_class_percent', type=int, default=0.01,
                    help='percent of val sample for per class (default: 50)')

parser.add_argument('--load_data_type', type=int, default=BY_N_PER_CLASS,
                    help='load dat type '
                         'BY_N_PER_CLASS --> 按照个数 / '
                         'BY_PERCENT_PER_CLASS --> 按半分比 (default: 50)')

parser.add_argument('--lr', type=float, default=0.0005, metavar='N',
                    help='leraning rate (default: 0.003)')

parser.add_argument('--lr_max', type=float, default=0.001, metavar='N',
                    help='max leraning rate (default: 0.001)')

parser.add_argument('--lr_min', type=float, default=0.0001, metavar='N',
                    help='min leraning rate (default: 0.00005)')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size  (default: 64)')

parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='train epoch (default: 100)')

parser.add_argument('--log_epoch', type=int, default=5, metavar='N',
                    help='train log epoch (default: 5)')

parser.add_argument('--best_model_path', type=str, default=r'.\实验结果',
                    help='file path to the model (default: none)')

parser.add_argument('--debug', type=bool, default=False,
                    help='debug or not (default: Flase)')

parser.add_argument('--cuda', type=bool, default=True,
                    help='cuda is available or not (default: False)')


parser.add_argument('--log_path', type=str, default=r'.\log',
                    help='file path to the model (default: none)')

args = parser.parse_args()

if __name__ == "__main__":
    " [日志设置] "
    args.time = time.strftime("%Y-%m-%d-%H-%M-%S") # 年月日作为log文件名
    args.log_file_path = "{}\{}-log.txt".format(args.log_path, args.time)

    args.logger = logger_config(log_path=args.log_file_path, logging_name='Siamese network')

    args.logger.info("---------------- 超参数记录 ----------------")
    args.logger.info("batch size ：{}".format(args.batch_size))
    args.logger.info("training epoch ：{}".format(args.epoch))
    args.logger.info("learning rate ：{}".format(args.lr))
    args.logger.info("max learning rate ：{}".format(args.lr_max))
    args.logger.info("min learning rate ：{}".format(args.lr_min))
    args.logger.info("train percent per class ：{}".format(args.train_class_percent))
    args.log_interval = 1000

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.log_file_path):
        os.system(r'touch %s' % args.log_file_path)

    args.cuda = torch.cuda.is_available()
    args.time = 10
    args.need_train = False  # 是否需要训练
    oa_arr = []
    aa_mean_arr = []
    aa_arr = []
    kappa_arr = []
    for i in range(args.time):

        data_Loader = dataLoader(args)

        # args.train_dataloader, args.test_dataloader = dataLoader.load_isar_data()
        args.train_dataloader, args.test_dataloader = data_Loader.load_ship_data()

        args.x_train_num = data_Loader.x_train_num
        args.x_val_num = data_Loader.x_val_num
        args.x_test_num = data_Loader.x_test_num
        args.y_train = data_Loader.y_train
        args.y_val = data_Loader.y_val
        args.y_test = data_Loader.y_test

        args.n_class = data_Loader.n_class

        # args.train_dataloader, args.val_dataloader, args.test_dataloader = dataLoader.MNIST_data_loader(mean = 0.1307, std = 0.3081)

        if(args.need_train):
            train(args)

        OA, AA_mean, Kappa, AA, matrix = test(args)
        oa_arr.append(OA)
        aa_mean_arr.append(AA_mean)
        aa_arr.append(AA)
        kappa_arr.append(Kappa)
        print(matrix)

    print("OA arr : ", oa_arr)
    print('OA均值 = ', np.mean(oa_arr))
    print('OA方差 = ', np.std(oa_arr))
    print('AA均值 = ', np.mean(aa_mean_arr, axis=0))
    print('AA方差 = ', np.std(aa_mean_arr, axis=0))
    print('kappa均值:', np.mean(kappa_arr))
    print('kappa方差:', np.std(kappa_arr))
    print('每类分类精度均值 = ', np.mean(aa_arr, axis=0))
    print('每类分类精度方差 = ', np.std(aa_arr, axis=0))

