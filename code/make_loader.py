import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题

class DealDataset(Dataset):

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.len = len(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def np_move_avg(a,n,mode="same"):#均值滤波

    return(np.convolve(a, np.ones((n,))/n, mode=mode))

def make_loader(seqLen,batch_size,data_path,y_col,pre_length):

    #pre_length = 1

    #file_name = '../data/temperature_merge.csv'
    file_name = data_path
    tem_data = pd.read_csv(file_name,encoding='GBK')
    data_label = tem_data[y_col].values
    tem_data = tem_data.drop(y_col,axis=1).values
    #data_1 = tem_data['推力轴瓦1温度'][4200:].values
    train_length = int(0.8*(len(tem_data)))
    data_train = tem_data[:train_length,:]
    data_test = tem_data[train_length:,:]
    label_train = data_label[:train_length]
    label_test = data_label[train_length:]
    # data_1 = np_move_avg(data_1, filter_size)
    # data_1 = data_1[filter_size:-filter_size]
    #plt.plot(range(len(data_1)),data_1)
    #plt.show()
    if tem_data.shape[1] == 1:
        data_train = data_train.reshape(-1,1)
        data_test = data_test.reshape(-1,1)
    stan = StandardScaler().fit(data_train)
    data_1_train = stan.transform(data_train)
    data_1_test = stan.transform(data_test)
    x_train = []
    y_train = []
    for i in range(seqLen, len(data_1_train)):
        x_train.append(data_1_train[i - seqLen:i])
        y_train.append(label_train[i:i+pre_length])
    # labels_test = torch.Tensor(y_val[seqLen:])
    # labels = labels.to(torch.double)
    x_test = []
    y_test = []
    for i in range(seqLen, len(data_1_test),pre_length):
        x_test.append(data_1_test[i - seqLen:i])
        y_test.append(label_test[i:i+pre_length])


    dealDataset = DealDataset(x_train, y_train)
    dealDataset1 = DealDataset(x_test, y_test)
    # 每次导入batch_size个序列给训练器，shuffle表示是否打乱批数据
    train_loader = DataLoader(dataset=dealDataset, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(dataset=dealDataset1, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader,test_loader,stan,data_1_test