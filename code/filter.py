import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题
file_name = 'merge_new.csv'
tem_data = pd.read_csv(file_name,encoding='GBK')
#tem_data.plot()
#plt.show()
data_1 = tem_data['value推力轴瓦'][4200:].values
#filter1 = np.array([1/3,1/3,1/3])
data_2 = tem_data['value上导轴瓦']
def np_move_avg(a,n,mode="same"):

    return(np.convolve(a, np.ones((n,))/n, mode=mode))
filter_size = 10
data_1_new = np_move_avg(data_1,filter_size)
plt.plot(range(len(data_1_new)),data_1_new,label = '均值滤波')
#plt.plot(range(len(data_1)),data_1)
#plt.show()

#Median filtering of one-dimensional time series
def median_filter(data,filter_size):
    data_new = []
    for i in range(len(data)):
        if i < filter_size:
            data_new.append(np.median(data[:i+1]))
        else:
            data_new.append(np.median(data[i-filter_size:i+1]))
    return data_new 

data_1_new1 = median_filter(data_1,filter_size)
plt.plot(range(len(data_1_new1)),data_1_new1, label = '中值滤波')
plt.plot(range(len(data_1)),data_1 , label = '原数据')
plt.legend()
plt.show()