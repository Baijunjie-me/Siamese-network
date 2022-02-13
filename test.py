# ----------------------------------测试样本集------------------------------------------ #
import torch
import SiameseNet
import numpy as np
cuda = torch.cuda.is_available()
cuda = False
torch.backends.cudnn.enabled = False
# ------------------------------------ 测试过程 -------------------------------------- #
def test(args):
    margin = 1.
    loss_fn = SiameseNet.ContrastiveLoss(margin)

    for epoch in range(0,1):
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
            pred = np.argmax(pred_arr, axis=1)

            print('Epoch: ', epoch,'batch:',batch_idx, '| test loss: %.4f' % total_loss)
        acc = len(pred == args.y_test)
        print("acc {} ".format((acc / len(args.y_test))))