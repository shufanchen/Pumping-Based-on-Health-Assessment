import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def merge_data(path, data_type):
    file_name = []
    listdir(path, file_name)
    merge_file = pd.DataFrame()
    for i in range(len(file_name)):
        if data_type == 'temperature':
            df = pd.read_csv(file_name[i])
            df = df[['monitor_timestamp', 'monitor_value']]
            df = df.rename(
                columns={'monitor_timestamp': 'time', 'monitor_value': file_name[i].split('.')[-2].split('\\')[-1]})
        elif data_type == 'vibrate':
            df = pd.read_excel(file_name[i])
            df = df.rename(columns={'时间': 'time', '监测值': file_name[i].split('.')[-2].split('\\')[-1]})
        df = df.set_index('time')
        if i == 0:
            merge_file = df
        else:
            merge_file = pd.concat([merge_file, df], axis=1, join='inner')
    file_path = '../output/' + data_type + '_merge.csv'
    #merge_file.to_csv(file_path, index=True, encoding='GBK')
    return file_path,merge_file


def box_plot(data):
    data.boxplot()
    plt.show()
    return


def minor_delete(data, thr=0.3): #thr代表删除阈值，低于这个阈值的数据点都要删除
    data = data
    col_ori = list(data.columns)
    col_new = []
    for i in range(len(col_ori)):
        col_new.append(chr(i + 65))
    data.columns = col_new
    data = data.drop(index=data[(data.B <= thr)].index.tolist())
    data.columns = col_ori
    # data.plot()
    # plt.show()
    return data


def fill_na(data):
    nan_model = SimpleImputer(missing_values=np.nan, strategy='mean')  # 建立替换规则：将值为NaN的缺失值以均值做替换
    data = nan_model.fit_transform(data)  # 应用模型规则
    return data


def stand_trans(data_0, data_1):
    stan = StandardScaler().fit(data_0)
    result = stan.transform(data_1)
    return result

if __name__ == '__main__':
    file_path,merge_d = merge_data('../data1/','vibrate')
    # plt.figure(1)
    # merge_d.plot()
    # plt.show()
    merge_d = minor_delete(merge_d)
    # plt.figure(2)
    # merge_d.plot()
    #plt.show()
    # box_plot(merge_d)
    # #plt.figure(3)
    # merge_d.plot()
    # merge_d.boxplot()
    # plt.show()
    # C=1
    merge_d.to_csv(file_path, index=True, encoding='GBK')
