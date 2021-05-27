# -*- coding: utf-8 -*-
"""
Created on Thu May 20 00:12:50 2021

@author: user
"""

import os
import numpy as np
import PIL
import torch
import pandas as pd
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
import csv

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_name('efficientnet-b3')
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


batch_size = 4

test_data=MyDataset(csv_path='data/test.csv',file_path = 'data/test_images', transform=tfms)
data_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)


model = model.to(device)
model.load_state_dict(torch.load("net0.16_120.pth"))
use_gpu = torch.cuda.is_available()


label = []
data_num = len(test_data)
for j, valdata in enumerate(data_loader):
    x_val, y_val = valdata
    if use_gpu:
        x_val, y_val = Variable(x_val.cuda()),Variable(y_val.cuda())
        model = model.cuda()
    else:
        x_val, y_val = Variable(x_val),Variable(y_val)
    y_pre = model.forward(x_val)
    prob = F.softmax(y_pre,dim=1)
    _, label_pre = torch.max(y_pre.data, 1)
    label_batch = label_pre.cpu().detach().numpy()
    label.append(label_batch)

label_all = []
for i in range(len(label)):
    for z in label[i]:
        label_all.append(z)
        
data = pd.read_csv(r'test.csv')
data['Label'] = label_all
data.to_csv(r"test6.csv",mode = 'a',index =False)


# train_file = open('train_part.csv','x',newline='')
# train_write = csv.writer(train_file)
# train_write.writerow(['index']+['label'])
# for i in label_all:
#     train_write.writerow([rows_data[i][0]]+[str(rows_data[i][1])])
# train_file.close()
