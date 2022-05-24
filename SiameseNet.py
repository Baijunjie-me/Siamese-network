import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                                     nn.BatchNorm2d(32), nn.PReLU(),
                                     nn.Dropout(0.5),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                     nn.BatchNorm2d(64), nn.PReLU(),
                                     nn.Dropout(0.5),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                                     nn.MaxPool2d(2, stride=2), )

        self.fc = nn.Sequential(nn.Linear(4608, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, 512),
                                nn.PReLU(),
                                nn.Linear(512, 2),
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)










