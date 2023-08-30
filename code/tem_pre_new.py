import numpy as np
import matplotlib.pyplot as plt
from tem_pre import Model
from make_loader import *
import torch
lstm = torch.load('../model/temp_pre.pkl')
_,__,stan,data = make_loader()
seq_len = 200
pred_len = 5
begin_Data = data[:seq_len]
begin_list = list(begin_Data)
# begin_Data = begin_Data.reshape(1,50,1)
for i in range(pred_len):
    if i==0:
        begin_Data = begin_Data.reshape(1, seq_len, 1)
        begin_Data = torch.tensor(begin_Data)
        begin_Data = begin_Data.to(torch.float32)
        pred = lstm(begin_Data)
        begin_list.append(pred.detach().numpy())
    else:
        begin_list.pop(0)
        begin_Data = torch.tensor(begin_list)
        begin_Data = begin_Data.to(torch.float32)
        begin_Data = begin_Data.reshape(1,seq_len,1)
        pred = lstm(begin_Data)
        begin_list.append(pred.detach().numpy())
labels = data[seq_len:(seq_len+pred_len)]
preds = begin_list[-pred_len:]
preds = np.array(preds)
preds = preds.reshape(pred_len,1)
labels = stan.inverse_transform(labels)
preds = stan.inverse_transform(preds)
plt.plot(range(pred_len),labels,label = 'facts')
plt.plot(range(pred_len),preds,label = 'preds')
plt.legend()
plt.show()

