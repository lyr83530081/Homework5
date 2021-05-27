# -*- coding: utf-8 -*-
"""
Created on Mon May 17 07:08:18 2021

@author: user
"""

import os
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn #神經網絡庫
import torch.nn.functional as F #內置函數庫
from torch import optim
import pickle
import cv2
import matplotlib.pyplot as plt
from MyDatasets import MyDataset
import time
from torch.autograd import Variable

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = EfficientNet.from_name('efficientnet-b0')
tfms = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


batch_size = 16
learning_rate = 0.0001

train_data=MyDataset(csv_path='data/train_part.csv',file_path = 'data/train_images', transform=tfms)
val_data=MyDataset(csv_path='data/val_part.csv', file_path = 'data/train_images', transform=tfms)
data_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
data_loader_val = DataLoader(val_data, batch_size=batch_size,shuffle=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    lr = learning_rate * (0.1 ** (epoch // 15))
    if lr >= 0.0001:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

len(train_data)

epoches = 50
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
use_gpu = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss(reduction='sum')
loss_train = []
loss_val = []
acc_list_val = []

print(len(data_loader))
for k in range(epoches):
    #adjust_learning_rate(optimizer,k)
    loss_t = 0
    print("-------------------------------------")
    print("epoch"+str(k))
    for i, traindata in enumerate(data_loader):
        x_train, y_train = traindata
        if use_gpu:
            x_train, y_train = Variable(x_train.cuda()),Variable(y_train.cuda())
            model = model.cuda()
        else:
            x_train, y_train = Variable(x_train),Variable(y_train)
        optimizer.zero_grad()
        y_pre = model.forward(x_train)
        prob = F.softmax(y_pre,dim=1)
        loss = criterion(y_pre, y_train)
        loss.backward()
        optimizer.step()
        aaa=loss.cpu().detach().numpy()
        loss_t+=aaa
        #print(aaa)
    loss_train.append(loss_t/len(data_loader))
    print(" Loss_val= {:.5}".format(loss_t/len(data_loader)))
    
    loss_v = 0
    correct_val = 0
    total = 0
    for j, valdata in enumerate(data_loader_val):
        x_val, y_val = valdata
        if use_gpu:
            x_val, y_val = Variable(x_val.cuda()),Variable(y_val.cuda())
            model = model.cuda()
        else:
            x_val, y_val = Variable(x_val),Variable(y_val)
        y_pre = model.forward(x_val)
        prob = F.softmax(y_pre,dim=1)
        loss = criterion(y_pre, y_val)
        aaa=loss.cpu().detach().numpy()
        loss_v += aaa
        _, label_pre = torch.max(y_pre.data, 1)
        correct_val += (label_pre == y_val.data).sum()
        total += y_val.size(0)
    loss_val.append(loss_v/len(data_loader_val))
    b = correct_val.cpu().detach().numpy()
    acc_list_val.append(b/len(data_loader_val))
    print(" Loss_val= {:.5},acc_val = {:.5}%".format(loss_v/len(data_loader_val),
                                                      100*b/(len(data_loader_val)*batch_size)))
    torch.save(model.state_dict(), 'checkpoints/net%.2f_%03d.pth' % (loss_v/len(data_loader_val),k))

plt.plot(loss_train,label='Loss train')
plt.plot(loss_val,label='Loss validation')
plt.legend() 
plt.show() 

plt.plot(acc_list_val,label='Accuracy')

plt.legend() 
plt.show()


