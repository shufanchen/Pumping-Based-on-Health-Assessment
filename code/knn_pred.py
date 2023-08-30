from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#from add_label import add_data_label
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题


def acc_cal(y_pre,y_facts):
    counts = 0
    for i in range(len(y_pre)):
        if y_pre[i] == y_facts[i]:
            counts = counts+1
    return counts/len(y_pre)*100

def build_knn(neighbor,reference,data,train_flag=1):
    kNN_reg = KNeighborsClassifier(n_neighbors=neighbor)
    #train = pd.read_csv('../data/vibrate_train_gs.csv',encoding='GBK')
    train = pd.read_csv(reference,encoding='GBK')
    x_train = train.iloc[:,1:-1].values
    y_train = train.iloc[:,[-1]].values
    #test = pd.read_csv('../data/vibrate_test_gs.csv', encoding='GBK')
    if train_flag:
        test = pd.read_csv(data, encoding='GBK')
        x_test = test.iloc[:, 1:-1].values
        y_test = test.iloc[:, [-1]].values
    else:
        x_test = pd.read_csv(data, encoding='GBK')
        x_test = x_test.iloc[:, 1:-1].values
    stan = StandardScaler().fit(x_train)
    x_train = stan.transform(x_train)
    x_test = stan.transform(x_test)
    kNN_reg.fit(x_train, y_train)
    Y_pre = kNN_reg.predict(x_test)
    plt.plot(range(len(Y_pre)), Y_pre,'*',label = '预测',markevery=5)
    if train_flag:
        plt.plot(range(len(y_test)), y_test,'o',label = '标签',markevery=5)
    plt.legend()
    plt.show()
    if train_flag:
        y_test = np.array(y_test)
        Y_val = y_test
        acc = acc_cal(Y_pre,Y_val)
        print(acc)
    else:
        return list(Y_pre).count(1),list(Y_pre).count(-1)


#build_knn(5)