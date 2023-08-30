import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题
from sklearn.preprocessing import StandardScaler

class PretrainDataset(Dataset):

    def __init__(self,
                 data,
                 sigma,
                 p=0.5,
                 multiplier=10):
        super().__init__()
        self.data = data
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape # num_ts, time, dim

    def __getitem__(self, item):
        ts = self.data[item % self.N]#从第一个维度随便拿一条序列进行增强
        return self.transform(ts)
    #self.transform(ts).shape = (4320,14)一条序列的长度

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):\
        #抖动，一种数据增强的方法
        if random.random() > self.p:
            return x + (torch.ones(x.shape) * self.sigma)
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)

class DealDataset(Dataset):

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.len = len(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

seqlen = 20
batch_size = 64
test_rate = 0.2
device = 'cuda'
def add_data_label(file_name='../data/vibrate_merge.csv'):
    raw_data = pd.read_csv(file_name,encoding='GBK')
    raw_data = raw_data.iloc[1:,1:]
    x = raw_data.values
    x_normal = x.reshape(1,-1,4)

    train_add = PretrainDataset(torch.from_numpy(x_normal).to(torch.float), sigma=0.05, multiplier=2)
    #数据增强，生成一些故障数据
    fault_Data = train_add[2023].cpu().numpy().reshape(406,4)
    fault_Data = pd.DataFrame(data=fault_Data,columns=raw_data.columns)
    plt.figure(1)
    raw_data.plot()

    plt.figure(2)
    fault_Data.plot()
    plt.show()

    y_fault = []

    for i in range(len(fault_Data)):
        y_fault.append(-1)
    fault_Data['label'] = y_fault
    test_length = int(test_rate*len(fault_Data))

    y_normal = []
    for i in range(len(x)):
        y_normal.append(1)
    raw_data['label'] = y_normal
    train_data = pd.concat([raw_data.iloc[test_length:,:],fault_Data.iloc[test_length:,:]])
    test_data  = pd.concat([raw_data.iloc[:test_length,:],fault_Data.iloc[:test_length,:]])
    train_data.to_csv('../data/vibrate_train_gs.csv', index=True, encoding='GBK')
    test_data.to_csv('../data/vibrate_test_gs.csv', index=True, encoding='GBK')
    return train_data,test_data


add_data_label()
C=1

# train_x = np.concatenate([fault_Data_1,x],axis=0)
# train_x = np.concatenate([train_x,fault_Data_2],axis=0)
# C = 1
# C=1
# for batch in train_loader:
#     fault_Data = batch.cpu().numpy().reshape(-1,4)
#     c = 1
