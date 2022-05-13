import os
percen_train_per_class_arr = [0.05, 0.1, 0.2]

lr_max_rate_arr = [0.001, 0.003, 0.005]
lr_min_rate_arr = [0.00003, 0.00005]
epoch_arr = [100, 150]
batch_size_arr = [512]
log_epoch = 10


# for _, percent in enumerate(percen_train_per_class_arr):
#     for _, lr in enumerate(lr_rate_arr):
#         for _, batch_size in enumerate(batch_size_arr):
#             for _, epoch, in enumerate(epoch_arr):
#                 try:
#                     os.system("python main.py --train_class_percent {} "
#                                   "               --lr {}"
#                                   "               --batch_size {}"
#                                   "               --epoch {} ".format(percent, lr, batch_size, epoch))
#                 except:
#                     raise RuntimeError('代码错了')

for _, percent in enumerate(percen_train_per_class_arr):
    for _, lr_max in enumerate(lr_max_rate_arr):
        for _, lr_min in enumerate(lr_min_rate_arr):

            for _, batch_size in enumerate(batch_size_arr):
                for _, epoch, in enumerate(epoch_arr):
                    try:
                        os.system("python main.py --train_class_percent {} "
                                      "               --lr_max {}"
                                      "               --lr_min {}"
                                      "               --batch_size {}"
                                      "               --epoch {} "
                                      "               --log_epoch {} ".format(percent, lr_max, lr_min, batch_size, epoch, log_epoch))
                    except:
                        raise RuntimeError('代码错了')