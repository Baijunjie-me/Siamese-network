# -------------------------------------------------用于数据的训练---------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import SiameseMNIST
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import SiameseNet
cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = False  # cudnn的两个必须都为true才可以
# torch.backends.cudnn.benchmark = True

# 数据导入
mean, std = 0.1307, 0.3081

n_classes = 10
siamese_train_dataset = SiameseMNIST(train_dataset)  # 返回图片对 与是否属于同一类的信息


# --------------------------------------------------训练---------------------------------------------------- #
def train(args):
    # batch_size = 64
    # siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=args.batch_size, shuffle=True)
    # 数据集
    train_dataset = MNIST(args.data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))
                          ]))
    margin = 1.
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = ContrastiveLoss(margin)
    # lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    # args.epoch = 20
    log_interval = 100

    # ------------------------------------------------训练过程----------------------------------------------------------#
    for epoch in range(0, args.epoch):
        for batch_idx, (data, target) in enumerate(args.train_dataloader):
            # print("data",data)
            # print("target",target)

            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            optimizer.zero_grad()
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            # losses.append(loss.item())
            # total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                total_loss = loss.item()
                print('Epoch: ', epoch, 'batch:', batch_idx, '| train loss: %.4f' % total_loss)