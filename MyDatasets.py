# -*- coding: utf-8 -*-
"""
Created on Mon May 17 07:49:54 2021

@author: user
"""

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import csv
import math
from itertools import islice

# csv_train = 'data/train.csv'

# rows_data = []
# sample_rate_val = 0.1 
# with open(csv_train,'r') as r_file:
#     file_read = csv.reader(r_file)
#     for row in islice(file_read,1,None):
#         #print(1)
#         rows_data.append((row[0],int(row[1])))
# r_file.close()
        
# data_num = len(rows_data)
# val_num = math.floor(data_num*sample_rate_val)  
# L = math.floor(data_num/val_num)
   
# train_file = open('data/train_part.csv','w',newline='')
# train_write = csv.writer(train_file)
# train_write.writerow(['index']+['label'])
# for i in range(data_num):
#     if (i+1)%L != 0:
#         train_write.writerow([rows_data[i][0]]+[str(rows_data[i][1])])
# train_file.close()

# val_file = open('data/val_part.csv','w',newline='')
# val_write = csv.writer(val_file)
# val_write.writerow(['index']+['label'])
# for i in range(val_num):
#     k = (i+1)*L-1
#     print(k,rows_data[k][0],rows_data[k][1])
#     val_write.writerow([rows_data[k][0]]+[str(rows_data[k][1])])
# val_file.close()

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, csv_path,file_path, transform=None, target_transform=None, loader=default_loader):
        r_file = open(csv_path,'r')
        file_read = csv.reader(r_file)
        rows = []
        for row in islice(file_read,1,None):
            if row[1] == '':
                temp = -1
                rows.append((row[0],temp))
            else:
                rows.append((row[0],int(row[1])))
        self.file_path = file_path
        self.rows = rows
        self.transform = transform

        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.rows[index]
        img = self.loader(self.file_path+'/'+fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.rows)







