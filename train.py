# -------------------------------------------------用于数据的训练---------------------------------------------------
from torch.optim import lr_scheduler
import numpy as np
from SiameseNet import *
import torch
torch.backends.cudnn.enabled = False  # cudnn的两个必须都为true才可以
# torch.backends.cudnn.benchmark = True

# --------------------------------------------------训练---------------------------------------------------- #
def train(args):
    margin = 1.
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if args.cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100

    # ------------------------------------------------训练过程----------------------------------------------------------#
    for epoch in range(0, args.epoch):
        pred_arr = np.zeros([args.x_train_num, args.n_class], dtype=np.float64)
        for batch_idx, (data0, label0, data1, label1, target, data0_idx) in enumerate(args.train_dataloader):
            target = target if len(target) > 0 else None
            if args.cuda:
                data0, label0, data1, label1, target = data0.cuda(), label0.cuda(), data1.cuda(), label1.cuda(), target.cuda()

            optimizer.zero_grad()
            outputs = model(data0, data1)
            # 计算相似度
            result = torch.abs(outputs[0]-outputs[1])  # 计算两个输出点之间的绝对值距离
            distance = result.sum(dim =1)

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


            if batch_idx % log_interval == 0:
                total_loss = loss.item()
                print('Epoch: ', epoch, 'batch:', batch_idx, '| train loss: %.4f' % total_loss)

        args.model = model
        if epoch == args.epoch - 1:
            # 最后一个epoch统计精读等信息
            pred = np.argmax(pred_arr, axis=1)
            acc = len(pred == args.y_train)


