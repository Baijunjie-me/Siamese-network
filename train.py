# -------------------------------------------------用于数据的训练---------------------------------------------------
from torch.optim import lr_scheduler

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
        for batch_idx, (data, target) in enumerate(args.train_dataloader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if args.cuda:
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

            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                total_loss = loss.item()
                print('Epoch: ', epoch, 'batch:', batch_idx, '| train loss: %.4f' % total_loss)