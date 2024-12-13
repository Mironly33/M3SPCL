import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import *
import torch
import random


def load_data(dataset,view_num):
    train_data = []
    label = []
    test_data =[]

    mat = sio.loadmat('./MVSPCL/datasets/' + dataset + '.mat')

  
    if dataset == 'Scene15':     #  X:4485*20/59/40   
        data = mat['X'][0][0:view_num]  # 20, 59 dimensions  
        label = np.squeeze(mat['Y'])
        divide_seed = 10


    if np.min(label) == 1:
        label = label - 1

    

    train_idx, test_idx = TT_split(len(label),  divide_seed)
    for i in range(len(data)):
        train_data.append(data[i][train_idx].T)
        test_data.append(data[i][test_idx].T)

    train_label, test_label = label[train_idx], label[test_idx]
    class_num = len(np.unique(label))
    dim_list = [data[i].shape[1] for i in range(len(data))]

    return train_data, train_label,test_data, test_label, divide_seed,class_num,dim_list


class GetDataset(Dataset):
    def __init__(self, data,label):
        self.data = [torch.from_numpy(view.astype(np.float32)).cuda() for view in data]
        self.label = torch.from_numpy(label).long().cuda() 

    def __getitem__(self, index):  
        
        X_list = [view[:, index].unsqueeze(0) for view in self.data] 
        label = self.label[index]
        return X_list, label
    def __len__(self):
        return len(self.label)


# 总！
def loader(train_bs, dataset,view_num):
    train_data, train_label,test_data, test_label, divide_seed,\
            class_num,dim_list= load_data(dataset,view_num)


    train_dataset = GetDataset(train_data,train_label)   

    test_dataset = GetDataset(test_data, test_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,  #1024
        shuffle=True,
        drop_last=True
    ) 
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )  
    return train_loader, test_loader, divide_seed,class_num,dim_list,test_data,test_label 
