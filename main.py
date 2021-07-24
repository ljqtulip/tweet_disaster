import pandas as pd
import torch.nn as nn
import torchtext.vocab as vocab
import torch
import os
from torch.utils.data import Dataset,DataLoader
import torchkeras
import numpy as np
max_len=209
BATCH_SIZE = 10

data_path = './data/'
train = pd.read_csv(data_path+'train_process.csv',index_col=0)
test = pd.read_csv(data_path+'test_process.csv',index_col=0)
test['target']=np.nan
glove = vocab.GloVe(name='6B', dim=100)

def text2ids(text):
    ids = []
    words = text.split(' ')
    for w in words:
        if w in glove.stoi.keys():
            ids.append(glove.stoi[w])
        elif w.lower() in glove.stoi.keys():
            ids.append(glove.stoi[w.lower()])
        else:
            ids.append(glove.stoi['unk'])
    tmp = [glove.stoi['pad']] * (max_len - len(ids))
    tmp.extend(ids)
    tmp = torch.Tensor(tmp).int()
    return tmp

class tweetDataset(Dataset):
    def __init__(self,data):
        # self.samples_path = samples_path
        # self.data = pd.read_csv(samples_path,index_col=0)
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        row = self.data.iloc[index]
        text = row['text']
        ids = text2ids(text)
        label = row['target']
        label = torch.tensor([float(label)],dtype = torch.float)
        return (ids,label)


# train_samples_path = data_path+'train_process.csv'
# test_samples_path = data_path+'test_process.csv'
ds_train = tweetDataset(train)
ds_test = tweetDataset(test)
dl_train = DataLoader(ds_train,batch_size = BATCH_SIZE,shuffle = True)
dl_test = DataLoader(ds_test,batch_size = BATCH_SIZE)

class Net(torchkeras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove.vectors))
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 100,out_channels = 128,kernel_size = 5))
        self.conv.add_module("pool_1", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("conv_2", nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5))
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())
        self.conv.add_module("conv_3", nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5))
        self.conv.add_module("pool_3", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_3", nn.ReLU())
        self.conv.add_module("conv_4", nn.Conv1d(in_channels=128, out_channels=16, kernel_size=5))
        self.conv.add_module("pool_4", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_4", nn.ReLU())


        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y

model = Net()
print(model)

model.summary(input_shape = (max_len,),input_dtype = torch.LongTensor)


def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc

model.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adagrad(model.parameters(),lr = 0.005),
             metrics_dict={"accuracy":accuracy})

dfhistory = model.fit(25,dl_train,log_step_freq= 10)

predict = model.predict(dl_test)
predict = predict.reshape(predict.shape[0])
test['target']=predict
for index, row in test.iterrows():
    target = row['target']
    if target > 0.5:
        test.at[index, 'target'] = 1
    else:
        test.at[index, 'target'] = 0
test.drop(['text','keyword','location'],axis=1,inplace=True)
test['target'] = test['target'].astype('int')
test.to_csv(data_path+'result_conv.csv')