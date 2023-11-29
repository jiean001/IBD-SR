# dataloader.py
import torch
import torch.nn.functional as F
import scipy.io as sio
import pickle
import numpy as np
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self):
        self.name = None
        self.num_target_class = None
        self.num_sensitive_class = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.data_path = None

        self.train_size = None
        self.test_size = None

        self.train_data = None
        self.train_label = None
        self.train_sensitive_label = None
        self.train_set = None

        self.test_data = None
        self.test_label = None
        self.test_sensitive_label = None
        self.test_set = None


class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor):
        super(MyDataset, self).__init__()
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.sensitive_tensor = sensitive_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class MyDatasetWithoutLabel(Dataset):
    def __init__(self, data_tensor):
        super(MyDatasetWithoutLabel, self).__init__()
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class SmileyDataset(BaseDataset):
    def __init__(self, dataset_args):
        super(SmileyDataset, self).__init__()
        self.name = "Smiley"
        self.s = None
        self.v = None
        self.LB = None

        self.num_target_class = None
        self.num_sensitive_class = None
        self.train_batch_size = dataset_args.train_batch_size
        self.test_batch_size = dataset_args.test_batch_size
        self.data_path = "{}/{}".format(dataset_args.dataroot, dataset_args.dataset)
        self.load()

    def load(self):
        train_dict = sio.loadmat(self.data_path + '/smiley_sparse_dataset_train_Alpha05.mat')
        train_data_unnorm = torch.from_numpy(train_dict['X']).float()
        self.train_data = F.normalize(train_data_unnorm, p=2, dim=1)

        test_dict = sio.loadmat(self.data_path + '/smiley_sparse_dataset_test_Alpha05.mat')
        test_data_unnorm = torch.from_numpy(test_dict['X']).float()
        self.test_data = F.normalize(test_data_unnorm, p=2, dim=1)

        self.s = test_dict['S']
        self.v = test_dict['V']
        self.LB = self.s * self.v  # the source vector, product of continuous variables V and binary activations S

        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

        self.train_set = MyDatasetWithoutLabel(self.train_data)
        self.test_set = MyDatasetWithoutLabel(self.test_data)


class ExtendedYaleBDataset(BaseDataset):
    def __init__(self, dataset_args):
        super(ExtendedYaleBDataset, self).__init__()
        self.name = "ExtendedYaleB"
        self.num_target_class = dataset_args.num_target_class
        self.num_sensitive_class = dataset_args.num_sensitive_class
        self.train_batch_size = dataset_args.train_batch_size
        self.test_batch_size = dataset_args.test_batch_size
        self.data_path = "{}/{}".format(dataset_args.dataroot, dataset_args.dataset)

        self.load()

    def load(self):
        data1 = pickle.load(open(self.data_path + "/set_0.pdata", "rb"), encoding='latin1')
        data2 = pickle.load(open(self.data_path + "/set_1.pdata", "rb"), encoding='latin1')
        data3 = pickle.load(open(self.data_path + "/set_2.pdata", "rb"), encoding='latin1')
        data4 = pickle.load(open(self.data_path + "/set_3.pdata", "rb"), encoding='latin1')
        data5 = pickle.load(open(self.data_path + "/set_4.pdata", "rb"), encoding='latin1')
        test = pickle.load(open(self.data_path + "/test.pdata", "rb"), encoding='latin1')

        train_data = np.concatenate(
            (data1["x"], data2["x"], data3["x"], data4["x"], data5["x"]), axis=0)
        train_label = np.concatenate(
            (data1["t"], data2["t"], data3["t"], data4["t"], data5["t"]), axis=0)
        train_sensitive_label = np.concatenate(
            (data1["light"], data2["light"], data3["light"], data4["light"], data5["light"]), axis=0)

        test_data = test["x"]
        test_label = test["t"]
        test_sensitive_label = test["light"]
        index = test_sensitive_label != 5
        test_label = test_label[index]
        test_sensitive_label = test_sensitive_label[index]
        test_data = test_data[index]

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]

        self.train_data = torch.from_numpy(train_data / 255.).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()
        self.train_set = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)

        self.test_data = torch.from_numpy(test_data / 255.).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()
        self.test_set = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)

