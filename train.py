# -------------------------------------------------用于数据的训练---------------------------------------------------
from torch.optim import lr_scheduler
import numpy as np
from SiameseNet import *
import torch
from test import test
from math import cos, pi
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

torch.backends.cudnn.enabled = False  # cudnn的两个必须都为true才可以
torch.backends.cudnn.benchmark = True

# ---------------------------------------------- T-SNE 降维可视化--------------------------------------------- #


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
# --------------------------------------------------训练---------------------------------------------------- #


def adjust_learning_rate(optimizer, current_epoch,max_epoch, lr_min=0,lr_max=0.1,warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * (current_epoch + 1) / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    margin = 1.
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if args.cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    args.log_interval = 1000
    lrs = [] # 记录每个epoch的学习率
    tsne = TSNE(n_components=2,init='pca',random_state=0)
    # ------------------------------------------------训练过程----------------------------------------------------------#
    for epoch in range(0, args.epoch):
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epoch, lr_min=args.lr_min,
                             lr_max=args.lr_max, warmup=False)
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        pred_arr = np.zeros([args.x_train_num, args.n_class], dtype=np.float64)
        for batch_idx, (data0, label0, data1, label1, target, data0_idx) in enumerate(args.train_dataloader):
            target = target if len(target) > 0 else None
            if args.cuda:
                data0, label0, data1, label1, target = data0.cuda(), label0.cuda(), data1.cuda(), label1.cuda(), target.cuda()

            optimizer.zero_grad()
            outputs = model(data0, data1)
            # 计算相似度
            result = torch.abs(outputs[0]-outputs[1])  # 计算两个输出点之间的绝对值距离
            distance = result.sum(dim=1)
            data0_idx = data0_idx.numpy()
            label1 = label1.cpu().numpy()
            distance = distance.detach().cpu().numpy()

            pred_arr[data0_idx, label1] += distance

            # 计算loss
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                total_loss = loss.item()
                args.logger.info('Epoch: {} batch {} | train loss {}, lr {} '.format(epoch,
                                                                                     batch_idx,
                                                                                     total_loss,
                                                                                     optimizer.param_groups[0]['lr']))
        args.model = model
        # 计算精度

        if epoch % args.log_epoch == 0:
            pred = np.argmin(pred_arr, axis=1)
            acc = np.sum(pred == args.y_train)
            args.logger.info("train acc {} ".format((acc / len(args.y_train))))
            test(args)


    torch.save(model.state_dict(), args.model_path)
