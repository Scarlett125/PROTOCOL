import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler
from im_dataset import get_imbalance_dataset

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
        self.data = [self.V1, self.V2]
        self.targets = self.Y 
    def __len__(self):
        return (len(self.targets))
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset,imbalance_ratio):   
    if dataset == "Hdigit":
        dataset = Hdigit('./data/')
        if imbalance_ratio != 1:
            im_dataset, im_data_size = get_imbalance_dataset(dataset,10,imbalance_ratio,split="train")
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    if imbalance_ratio == 1:
        return dataset, dims, view, data_size, class_num
    else:
        return im_dataset, dims, view, im_data_size, class_num
