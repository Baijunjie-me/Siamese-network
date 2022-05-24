# ----------------------------------测试样本集------------------------------------------ #
import torch
import SiameseNet
import numpy as np
from sklearn.metrics import confusion_matrix
import logging
from SiameseNet import *
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# ------------------------------------ 测试过程 -------------------------------------- #
def test(args):
    if not args.need_train:
        device = torch.device('cuda')
        embedding_net = EmbeddingNet()
        args.model = SiameseNet(embedding_net)
        args.model.load_state_dict(torch.load(args.model_path))
        args.model.to(device)
        # args.model.load(args.model_path, map_location=lambda storage, loc: storage.cuda(0))
        # torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda(0))
        # args.model.eval()
    margin = 1.
    loss_fn = ContrastiveLoss(margin)
    for epoch in range(0, 1):
        pred_arr = np.zeros([args.x_test_num, args.n_class], dtype=np.float64)
        for batch_idx, (data0, label0, data1, label1, target, data0_idx) in enumerate(args.test_dataloader):

            target = target if len(target) > 0 else None
            if args.cuda:
                data0, label0, data1, label1, target = data0.cuda(), label0.cuda(), data1.cuda(), label1.cuda(), target.cuda()

            outputs = args.model(data0, data1)
            # 计算相似度
            result = torch.abs(outputs[0] - outputs[1])
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

            total_loss = loss_outputs.item()

            if batch_idx % args.log_interval == 0:
                args.logger.info('test | batch idx {} loss {}'.format(batch_idx, total_loss))

        pred = np.argmin(pred_arr, axis=1)

        acc = np.sum(pred == args.y_test)
        args.logger.info("test acc {} ".format((acc / len(args.y_test))))
        matrix = confusion_matrix(args.y_test, pred)
        OA, AA_mean, Kappa, AA = cal_results(matrix)
        return OA, AA_mean, Kappa, AA, matrix
        # oa_arr.append(OA)
        # aa_mean_arr.append(AA_mean)
        # aa_arr.append(AA)
        # kappa_arr.append(Kappa)
        # print(matrix)
