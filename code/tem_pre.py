import json
from datetime import datetime
import os
import pandas as pd
import torch
from make_loader import make_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中汉字显示为方格的问题


class Model(torch.nn.Module):
    """LSTM预测模型本体"""

    def __init__(self, input_size, hidden_size, num_layers,pre_length):
        super(Model, self).__init__()
        #    self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=0.05)
        self.fc_1 = torch.nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = torch.nn.Linear(16, 8)  # fully connected 2
        self.fc = torch.nn.Linear(8, pre_length)  # fully connected last layer
        self.dropout = torch.nn.Dropout(0.01)

    def forward(self, input):  # input格式必须是(batch_size,seqLen,input_Size)
        out, (h_n, c_n) = self.rnn(input)
        hn_o = torch.Tensor(h_n.detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        out = self.fc_1(hn_o)
        out = self.fc_2(out)
        out = self.fc(out)
        return out


def print_model_params(model_name,params,train_set,val_loss,run_dir):
    """打印训练模型日志"""
    with open(run_dir+'/'+params['model_type']+'_model_training_log.txt', 'a') as f:
        f.write('/' + '*' * 80 + '/\n')
        f.write('训练日期:\t'+str(datetime.now())+'\n')
        f.write('模型名称:\t'+model_name+'\n')
        f.write('训练集名称:\t' + train_set + '\n')
        f.write('模型参数:\t')
        js = json.dumps(params)
        f.write(js)
        f.write('\n')
        f.write('val_loss:\t%.06f\n'% val_loss )
        f.write('/' + '*' * 80 + '/\n')
    return 0


def train_model(model_for_train, train_loader, criterion, optimizer,epochs):
    """训练模型"""
    for epoch in range(1, epochs):

        model_for_train.train()
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            try:
                inputs = inputs.to(torch.float32)
            except Exception:
                continue
            labels = labels.to(torch.float32)
            # print(inputs.shape)
            # inputs = torch.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
            prediction = model_for_train(inputs)
            try:
                loss = criterion(prediction, labels)
            except Exception:
                continue

            epoch_loss += loss.item()
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
        print(epoch_loss)
        if epoch_loss <= 0.003:
            break

    return


def testing_function(test_loader, model, stan):
    """测试模型函数，返回测试集对应的预测值"""
    result_test = list()
    label = []
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        label = label + list(labels.detach().numpy())
        inputs = inputs.to(torch.float32)
        prediction = model(inputs)
        prediction = prediction.detach().numpy()
        for j in prediction:
            # j = j.reshape(-1, 1)
            # j = stan.inverse_transform(j)
            result_test.append(j)
    return result_test, label

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def train_con(database_name_train,params,save_path,y_col):

    if params['model_type'] == 'LSTM':
        seqLen = params['seqLen']
        batch_size = params['batch_size']
        #input_size = params['input_size']
        lr = params['learning_rate']
        epochs = params['epochs']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        pre_length = params['pre_length']



    LR = lr

    train_loader, test_loader, stan, data_test = make_loader(seqLen,batch_size,database_name_train,y_col,pre_length)
    input_size = data_test.shape[1]
    params['input_size'] = input_size
    model = Model(input_size, hidden_size, num_layers,pre_length)

    #model = Model(input_size, N_HIDDEN, N_LAYER)  # our lstm class
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # model = torch.load('D:/PHM_code/低压电气开关寿命预测/模型保存/a1训练得到的模型/a1_database.pkl')
    train_model(model, train_loader, criterion, optimizer,epochs)

    pred, facts = testing_function(test_loader, model, stan)
    pred = np.array(pred)
    pred = pred.reshape((-1,))
    # labels = []
    # for f in facts:
    #     f = f.reshape(-1, 1)
    #     f = stan.inverse_transform(f)
    #     labels.append(f)
    facts = np.array(facts)
    facts = facts.reshape((-1,))
    # facts = stan.inverse_transform(facts)
    val_loss = torch.sqrt(criterion(torch.Tensor(pred), torch.Tensor(facts.reshape((-1,)))))
    plt.plot(range(len(pred)), pred, label='预测')
    plt.plot(range(len(pred)), facts, label='标签')
    plt.legend()
    plt.show()
    run_dir = f"../model/{database_name_train.split('/')[-1]}/{name_with_datetime('train')}"

    os.makedirs(run_dir, exist_ok=True)

    print_model_params('LSTM',params,database_name_train,val_loss,run_dir)
    if save_path == '0':
        torch.save(model, run_dir + '/model.pkl')
    else:
        torch.save(model, save_path+'/model.pkl')

def pred(data_path,model_name):
    data = pd.read_csv(data_path,encoding='GBK')
    #data = data['monitor_value'].values
    data = data.values
    data = data[-30:,:-1]
    stan = StandardScaler().fit(data)

    data = stan.transform(data)
    data = data.reshape(1, 30, 3)
    lstm = torch.load(model_name)
    prediction = lstm(torch.Tensor(data))
    return prediction
