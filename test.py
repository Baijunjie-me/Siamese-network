# ----------------------------------测试样本集------------------------------------------ #
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import SiameseMNIST
import torch
import SiameseNet
from torch.optim import lr_scheduler
import torch.optim as optim
cuda = torch.cuda.is_available()
cuda = False
torch.backends.cudnn.enabled = False
# ------------------------------------ 测试过程 -------------------------------------- #
def test(args):
    test_dataset = MNIST(args.data_path, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))
                         ]))
    for epoch in range(0,1):
        for batch_idx,(data,target) in enumerate(args.test_loader):
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            outputs=model(*data)
            # loss.backward()
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)

            if batch_idx % log_interval == 0:
                 total_loss=loss.item()
                 print('Epoch: ', epoch,'batch:',batch_idx, '| test loss: %.4f' % total_loss)