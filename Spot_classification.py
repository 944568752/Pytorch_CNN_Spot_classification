# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 23:38:14 2021

@author: 94456
"""

# Spot classification


import warnings
warnings.filterwarnings('ignore')


import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import torchvision.transforms.functional as tf


# Parameters
data_path=r'./Spot_dataset'
image_size=64
batch_size=4
Epoch=20



class Spot_dataset(Dataset):
    
    # mode : Train/Test 
    
    def __init__(self,data_path,image_size=64,mode='Train'):
        super(Spot_dataset,self).__init__()
        self.data_path=os.path.join(data_path,mode)
        self.image_size=image_size
        
        positive_path=glob.glob(os.path.join(self.data_path,'positive','*.jpg'))
        negative_path=glob.glob(os.path.join(self.data_path,'negative','*.jpg'))
        
        self.All_image_path=positive_path+negative_path
        self.All_label=[1]*len(positive_path)+[0]*len(negative_path)
        
    def __getitem__(self,item):
        input_image=Image.open(self.All_image_path[item]).convert('L')
        input_image=input_image.resize((self.image_size,self.image_size))
        input_label=self.All_label[item]
        
        # Data enhancement
        if random.random()>0.5:
            input_image=tf.hflip(input_image)
        if random.random()>0.5:
            input_image=tf.vflip(input_image)
        input_image=tf.to_tensor(input_image)
        input_label=torch.tensor(input_label,dtype=torch.int64)
        
        return input_image,input_label
    
    def __len__(self):
        return len(self.All_image_path)
    
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.conv_0=nn.Conv2d(1,6,5,padding=2,stride=1)
        self.bn_0=nn.BatchNorm2d(6)
        self.pool_0=nn.MaxPool2d(2,2)
        self.conv_1=nn.Conv2d(6,16,5,padding=2,stride=1)
        self.bn_1=nn.BatchNorm2d(16)
        self.pool_1=nn.MaxPool2d(2,2)
        self.conv_2=nn.Conv2d(16,32,5,padding=2,stride=1)
        self.bn_2=nn.BatchNorm2d(32)
        self.pool_2=nn.MaxPool2d(2,2)
        self.fc_0=nn.Linear(8*8*32,32)
        self.fc_1=nn.Linear(32,2)
        self.softmax=nn.Softmax()
        
    def forward(self,x):
        x=self.pool_0(F.relu(self.bn_0(self.conv_0(x))))
        x=self.pool_1(F.relu(self.bn_1(self.conv_1(x))))
        x=self.pool_2(F.relu(self.bn_2(self.conv_2(x))))
        x=x.reshape(-1,8*8*32)
        x=F.relu(self.fc_0(x))
        x=F.relu(self.fc_1(x))
        x=self.softmax(x)
        
        return x
    
    
cnn=CNN()


criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(cnn.parameters(),lr=0.0001,momentum=0.9)


# Train
trainloader=DataLoader(
                 Spot_dataset(data_path,image_size,mode='Train'),
                 batch_size=batch_size,
                 shuffle=True,
                 num_workers=0
            )

# Test
testloader=DataLoader(
                 Spot_dataset(data_path,image_size,mode='Test'),
                 batch_size=batch_size,
                 shuffle=False,
                 num_workers=0
            )


Loss_record=[]

for epoch in range(1,Epoch+1,1):
    # Train
    running_loss=0.0
    
    cnn.train()
    for inputs,labels in tqdm(trainloader,desc=f'Train in Epoch {epoch}/{Epoch}',total=len(trainloader)):
        
        optimizer.zero_grad()
        outputs=cnn(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss=running_loss+loss.item()  
    running_loss=running_loss/len(trainloader)
    Loss_record.append(running_loss)
    print(f'Train loss : {running_loss:.5f}',end='\n\n')
    
    
    # Test
    correct=0
    
    cnn.eval()
    with torch.no_grad():
        for inputs,labels in tqdm(testloader,desc=f'Test in Epoch {epoch}/{Epoch}',total=len(testloader)):
            outputs=cnn(inputs)        
            _,predicted=torch.max(outputs,dim=1)
            correct=correct+torch.sum(predicted==labels).item()
    print(f'Test accuracy : {correct}/18',end='\n\n')
            

# Show loss curve
plt.figure()
plt.title('Loss')
plt.plot(np.arange(1,Epoch+1,1),Loss_record,color='red',label='Training Loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.grid()
plt.show()
plt.clf()       
        
        
        
        
        
        
        