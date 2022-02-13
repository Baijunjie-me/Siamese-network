import os

from torch.utils import data

BY_N_PER_CLASS = 0
BY_PERCENT_PER_CLASS = 1
import scipy.misc as im
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as transforms
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class dataLoader(object):
    def __init__(self, args):
        self.model = None

        self.data_path = args.data_path

        self.train_batch_size = args.batch_size
        self.val_batch_size = args.batch_size
        self.test_batch_size = args.batch_size

        self.x_data = None
        self.y_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.load_data_type = args.load_data_type

        self.n_train_per_class = args.n_train_per_class
        self.n_val_per_class = args.n_val_per_class
        self.n_test_per_class = 0

        self.n_class = 0

        self.train_class_percent = args.train_class_percent
        self.val_class_percent = args.val_class_percent
        self.test_class_percent = 0
        
        self.shuffle_list = []

    def get_mstar_data(self, width=128, height=128, crop_size=28, aug=False):
        data_dir = self.data_path
        sub_dir = ["2S1", "BMP2", "BRDM_2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
        X = []
        y = []

        for i in range(len(sub_dir)):
            tmp_dir = data_dir + sub_dir[i] + "/"
            img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
            print(sub_dir[i], len(img_idx))
            y += [i] * len(img_idx)
            for j in range(len(img_idx)):
                img = im.imresize(im.imread((tmp_dir + img_idx[j])), [height, width])
                img = img[(height - crop_size) // 2: height - (height - crop_size) // 2, \
                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                # img = img[16:112, 16:112]   # crop
                # img = img[np.newaxis, :]
                X.append(img)
        
        return np.asarray(X), np.asarray(y)

    def data_shuffle(self, seed=0):
        X = self.x_data
        y = self.y_data
        data = np.hstack([X, y[:, np.newaxis]])
        np.random.shuffle(data)
        return data[:, :-1], data[:, -1]

    def cal_data_num(self, data_num_this_class):
        train_class_num = int(data_num_this_class * self.train_class_percent)
        val_class_num = int(data_num_this_class * self.val_class_percent)
        test_class_num = data_num_this_class - train_class_num - val_class_num
        return train_class_num, val_class_num, test_class_num

    def split_mstar_data(self):
        self.n_class = np.max(self.y_data)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = ([], []), ([], []), ([], [])
        for i in range(self.n_class):
            shuffle_list = list(np.where(self.y_data == i))[0]
            train_class_num, val_class_num, test_class_num = self.cal_data_num(np.size(shuffle_list))
            np.random.shuffle(shuffle_list)
            self.shuffle_list.append(shuffle_list)
            x_train.extend(self.x_data[shuffle_list[: train_class_num]])
            y_train.extend(self.y_data[shuffle_list[: train_class_num]])

            x_val.extend(self.x_data[shuffle_list[train_class_num : train_class_num + val_class_num]])
            y_val.extend(self.y_data[shuffle_list[train_class_num : train_class_num + val_class_num]])

            x_test.extend(self.x_data[shuffle_list[train_class_num + val_class_num :]])
            y_test.extend(self.y_data[shuffle_list[train_class_num + val_class_num :]])

        # x_train = np.reshape(x_train, (np.shape(x_train)[0], 1, np.shape(x_train)[1], np.shape(x_train)[2]))
        # x_val= np.reshape(x_val, (np.shape(x_val)[0], 1, np.shape(x_val)[1], np.shape(x_val)[2]))
        # x_test = np.reshape(x_test, (np.shape(x_test)[0], 1, np.shape(x_test)[1], np.shape(x_test)[2]))

        x_train = np.reshape(x_train, (np.shape(x_train)[0], 1, np.shape(x_train)[1], np.shape(x_train)[2]))
        x_val= np.reshape(x_val, (np.shape(x_val)[0], 1, np.shape(x_val)[1], np.shape(x_val)[2]))
        x_test = np.reshape(x_test, (np.shape(x_test)[0], 1, np.shape(x_test)[1], np.shape(x_test)[2]))

        print("x train shape : {}".format(np.shape(x_train)))
        print("x val shape : {}".format(np.shape(x_val)))
        print("x test shape : {}".format(np.shape(x_test)))

        self.x_train_num = np.shape(x_train)[0]
        self.x_val_num = np.shape(x_val)[0]
        self.x_test_num = np.shape(x_test)[0]

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test


        return (np.array(x_train).astype(np.float32), np.array(y_train).astype(np.uint8)), \
               (np.array(x_val).astype(np.float32), np.array(y_val).astype(np.uint8)), \
               (np.array(x_test).astype(np.float32), np.array(y_test).astype(np.uint8))

    def load_mstar_data(self):
        self.x_data, self.y_data = self.get_mstar_data()

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.split_mstar_data()
        
        print('data shape is {}'.format(np.shape(x_train)))
        
        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
        test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size)

        return self.train_loader, self.val_loader, self.test_loader

    def load_isar_data(self):
        self.x_data, self.y_data = self.get_mstar_data()

        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = self.split_mstar_data()

        print('data shape is {}'.format(np.shape(self.x_train)))

        train_dataset = self.create_dataset(self.x_train, self.y_train)
        test_dataset = self.create_dataset(self.x_test, self.y_test)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)

        return self.train_loader, self.test_loader

    def create_pairs(self, data, label):
        x0_data_list = []
        label0_list = []
        x1_data_list = []
        label1_list = []
        target_list = []
        data0_idx_list = []
        
        # 根据label索引训练集所有样本
        digit_indices_train = [np.where(self.y_train == i)[0] for i in range(self.n_class)]  # np.where()返   

        # 对于每一个样本和训练集的每个样本配对
        idx0 = 0
        for data0, label0 in zip(data, label):
            # 同类            
            same_class_idx = digit_indices_train[label0]

            for _, idx1 in enumerate(same_class_idx):
                if idx1 == idx0:
                    # 同类可能会和自己匹配到，idx相同时，continue
                    continue
                data1 = self.x_train[idx1]
                label1 = self.y_train[idx1]
                
                x0_data_list.append(data0)
                label0_list.append(label0)
            
                x1_data_list.append(data1)
                label1_list.append(label1)
                
                target_list.append(1) # 同类为1
                data0_idx_list.append(idx0)

            # 所有的异类匹配
            for label1 in range(self.n_class):
                if label1 == label0:
                    # 和自己的类别匹配时，continue
                    continue

                diff_class_idx = digit_indices_train[label1]

                for _, idx1 in enumerate(diff_class_idx):
    
                    data1 = self.x_train[idx1]
                    label1 = self.y_train[idx1]
                    
                    x0_data_list.append(data0)
                    label0_list.append(label0)
                
                    x1_data_list.append(data1)
                    label1_list.append(label1)
                    
                    target_list.append(0) # 异类为0
                    data0_idx_list.append(idx0)

            idx0 += 1

        x0_data = np.array(x0_data_list, dtype=np.float32)
        x0_data = x0_data.reshape([-1, 1, 28, 28])
        x1_data = np.array(x1_data_list, dtype=np.float32)
        x1_data = x1_data.reshape([-1, 1, 28, 28])
        label0 = np.array(label0_list, dtype=np.int32)
        label1 = np.array(label1_list, dtype=np.int32)
        target = np.array(target_list, dtype=np.int32)

        data0_idx = np.array(data0_idx_list, dtype=np.int32)

        return x0_data, label0, x1_data, label1, target, data0_idx
        
    def create_dataset(self, data, label):        
        x0, label0, x1, label1, target, data0_idx = self.create_pairs(data, label)
        
        ret = Dataset(x0, label0, x1, label1, target, data0_idx)
        return ret


    def MNIST_data_loader(self, mean = 0.1307, std = 0.3081):
        train_dataset = MNIST('../data/MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((mean,), (std,))
                              ]))
        test_dataset = MNIST('../data/MNIST', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))

        siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
        siamese_test_dataset = SiameseMNIST(test_dataset)
        # print("siamese",siamese_train_dataset[0:10])
        batch_size = 64
        # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True )
        siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False )
        # 测试集dataloader 做为 验证集dataloader
        return siamese_train_loader, siamese_test_loader, siamese_test_loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x0, label0, x1, label1, target, data0_idx):
        self.size = target.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.label0 = torch.from_numpy(label0)
        self.x1 = torch.from_numpy(x1)
        self.label1 = torch.from_numpy(label1)
        self.target = torch.from_numpy(target)
        self.data0_idx = torch.from_numpy(data0_idx)

    def __getitem__(self, index):
        return (self.x0[index],
                self.label0[index],
                self.x1[index],
                self.label1[index],
                self.target[index],
                self.data0_idx[index])

    def __len__(self):
        return self.size




class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}  # 标签到索引
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            # 构造同类样本集
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]
            # 构造不同类样本集
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)