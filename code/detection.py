import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题
def detection_abn(file):
    #data = pd.read_csv('../data/merge_new.csv',encoding='GBK')
    data = pd.read_csv(file,encoding='GBK')
    #data = data.iloc[:,[1]]
    # if data_type == 'temperature':
    #     data =  data.drop(index = data[(data.value排水总管 == 0)].index.tolist())
    # else:
    #     data = data.drop(index = data[(data.value下导Y <= 0.05)].index.tolist())
    df = data.iloc[:,1:]
    #构建模型 ,n_estimators=100 ,构建100颗树
    model = IsolationForest(n_estimators=100,
                          max_samples='auto',
                          contamination=float(0.01),
                          max_features=1.0)
    # 训练模型
    model.fit(df.values)

    # 预测 decision_function 可以得出 异常评分
    #df['scores']  = model.decision_function(df[['value排水总管']])
    data['scores']  = model.decision_function(df.values)
    #  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
    #data['anomaly'] = model.predict(df.values)
    data.plot()
    plt.show()
    #print(data['anomaly'].value_counts())

#detection_abn('../data/merge_new.csv')
#detection_abn('../data/vibrate_merge_new.csv')