# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:53:10 2024

CNN based on the LeNet structure: https://en.wikipedia.org/wiki/LeNet
Uses the handwritten character database from scikit learn, stored
in an intermediate text file.

This program features hand made convolution and pooling layers, with an
identical implementation using the native pytorch modules. Native pytorch
modules, in particular the convolutional layers, run faster than their
homebrew counterparts due to their C++ compilation.

An example run may produce:
    Step 2900:   train loss = 0.12462, test loss = 0.50112
    Confusion:
    tensor([[16.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0., 14.,  0.,  0.,  0.,  0.,  1.,  0.,  2.,  0.],
            [ 0.,  0., 16.,  5.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  9.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  0., 18.,  0.,  0.,  2.,  1.,  0.],
            [ 0.,  0.,  0.,  1.,  0., 10.,  0.,  0.,  0.,  2.],
            [ 0.,  0.,  0.,  0.,  0.,  0., 17.,  0.,  3.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 17.,  1.,  0.],
            [ 0.,  5.,  0.,  3.,  2.,  2.,  0.,  0., 10.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0., 15.]])
Note that the training loss is lower than the test loss, indicating model
overfitting. This is due to the small sample pool.



@author: James
"""

import torch
import torch.nn as nn
from torch.nn import functional as F 

#Hyper paramter set up -----------------------------------------------
device = 'cude' if torch.cuda.is_available() else 'cpu'
c1_kernel_size = 3
c1_kernel_count = 6
c2_kernel_size = 2
c2_kernel_count = 16 

p1_kernel_size = 2
p1_stride = 2
p2_kernel_size = 2
p2_stride = 2

dropout = 0.2

training_rounds = 3000
learning_rate = 3e-4
eval_it = 100 #Signposting during training
eval_counts = 100 #Sample count assessed during signposting
batch_size = 30

#Data input------------------------------------------------------------

f = open("digits.txt","r")
data = f.read()
data = data.split("\n")

images = data[::2]
for i in range(len(images)):
    images[i] = images[i].split(" ")
    for j in range(len(images[i])-1):
        images[i][j] = float(images[i][j])
    images[i].pop()
images.pop()

targets = data[1::2]
for i in range(len(targets)):
    targets[i] = int(targets[i])

#Images currently required to be square !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
image_size = int(len(images[0])**0.5)
target_size = len(set(targets))



#testTrain Split ----------------------------------------

data = torch.tensor(images[0]).view(1,8,8)
#data = (data - torch.mean(data))/torch.std(data) #Gaussian Normalisation
data = data/torch.max(data)                       #Range normalisation

for i in range(1,len(images)):
    new = torch.tensor(images[i]).view(1,8,8)
    new = new/torch.max(data)
    
    #data = (data - torch.mean(data))/torch.std(data)
    data = torch.cat((data,new),dim = 0)
del(new,images)

n = int(0.9*len(data))
train_data = data[:n]
test_data  = data[n:]
train_targets = torch.tensor(targets[:n])
test_targets = torch.tensor(targets[n:])
del(targets)

#Utility Functions------------------------------------------------------
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    targets = train_targets if split == 'train' else test_targets
    
    ix = torch.randint(len(data),(batch_size,))
    x = torch.stack([data[i].unsqueeze(0) for i in ix])
    y = torch.tensor([targets[i] for i in ix]).unsqueeze(1)
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad
def evaluate_loss():
    out = {}
    m.eval()
    
    for split in ['train','test']:
        losses = torch.zeros(eval_counts)
        for i in range(eval_counts):
            X,Y = get_batch(split)
            Y = Y.squeeze()
            logits = m(X)
            
            loss = F.cross_entropy(logits,Y)
            
            losses[i] = loss.item()
        out[split] = losses.mean()
        
    m.train()
    
    return out

#Layer definitions------------------------------------------------------
class ConvolutionN(nn.Module):
    def __init__(self,kernel_size,channel_in,padding = 0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(channel_in,kernel_size,kernel_size,)*(channel_in*kernel_size)**-0.5) #normalised by sqrt(total_parameter_count)
        self.bias = nn.Parameter(torch.randn(1)*((channel_in*kernel_size)**-0.5))
        self.padding = padding
        self.channel_in = channel_in
        
    def forward(self,x): 
        #Pad input
        x = F.pad(x,tuple([self.padding for _ in range(4)]),"constant",0)
        
        #Dims for ease of use, and output pre-allocation
        batch_size,channel_in,h,w = x.size() # B,C,H,W
        _,h_k,w_k = self.w.shape #H,W
        
        out_height = h-h_k+1
        out_width = w-w_k+1
        out = torch.zeros(batch_size,out_height,out_width)#B,H,W
        
        for i in range(out_height):
            for j in range(out_width):
                out[:,j,i] = torch.sum(x[:,:,j:j+h_k,i:i+w_k] * self.w,dim = (1,2,3)) + self.bias
        return out

class ConvolutionLayer(nn.Module):
    def __init__(self,kernel_size,channel_in,kernel_count,kernel_depth,padding = 0):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_count = kernel_count
        self.model = nn.ModuleList([ConvolutionN(kernel_size,channel_in,padding) for _ in range(kernel_count)])
    
    def forward(self,x):
        
        output = [kernel(x) for kernel in self.model]
        
        output = torch.stack(output,dim = 1)
        
        return output

class Pool(nn.Module):

    def __init__(self,kernel_size,stride,kernel_type = 'mean'):
        super().__init__()        
        self.stride = stride
        self.kernel_size = kernel_size
        
        #
        #Pool module currently cannot perform the max kernel
        #
        
        if kernel_type == 'mean':
            self.kernel = torch.ones(kernel_size,kernel_size)/kernel_size**2
    
    def forward(self,x):
        B,C,H,W = x.shape
        
        #Dimension check for stride compatibility
        if H//self.stride != H/self.stride or W//self.stride != W/self.stride:
            raise ValueError("Pooling layer dimension mismatch")
        
        #set up the output
        out = torch.zeros(B,C,H//self.stride,W//self.stride)
        
        for i in range(C):
            for row in range(0,H,self.stride):
                for col in range(0,W,self.stride):
                    out[:,i,row//self.stride,col//self.stride] = torch.sum(x[:,i,row:row+self.stride,col:col+self.stride]*self.kernel,dim = (-1,-2))
        return out

class ConvolutionNN(nn.Module):
         def __init__(self):
             super().__init__()
             self.model = nn.Sequential(
                 ConvolutionLayer(c1_kernel_size,1,c1_kernel_count,kernel_depth = 1,padding = 2),
                 nn.BatchNorm2d(6),
                 nn.ReLU(),
                 Pool(p1_kernel_size,stride = 2,kernel_type = 'mean'),
                 
                 ConvolutionLayer(c2_kernel_size,6,c2_kernel_count,kernel_depth = 1,padding = 0),
                 nn.BatchNorm2d(16),
                 nn.ReLU(),
                 Pool(p2_kernel_size,stride = 2,kernel_type = 'mean'),
                 
                 nn.Flatten(1,-1),
                 nn.Linear(64,30),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(30,15),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(15,target_size)
                 )
         
         def forward(self,x):
             for m in self.model:
                 x = m(x)
             return x
       
#LeNet realised with native pytorch modules for comparison:--------------
       
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=c1_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU())
        self.pu1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=c2_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pu2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.fc = nn.Linear(64, 30)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30, 15)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(15, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pu1(out)
        out = self.layer2(out)
        out = self.pu2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out        
       
        
       
#Model creation-------------------------------------------------------
m = ConvolutionNN()
# m = LeNet5(target_size)
m.to(device)
optimiser = torch.optim.Adam(m.parameters(),lr = learning_rate)

#Training loop--------------------------------------------------------
for i in range(training_rounds):
    
    xb,yb = get_batch('train')
    yb = yb.squeeze()
    
    logits = m(xb)
    
    loss = F.cross_entropy(logits,yb)
    
    optimiser.zero_grad(set_to_none = True)
    loss.backward() 
    optimiser.step()
    
    if i % eval_it == 0:
        losses = evaluate_loss()
        print("Step {}:   train loss = {:.5f}, test loss = {:.5f}".format(i,losses['train'],losses['test']))


#Confusion matrix of test_data-------------------------------------------

confusion = torch.zeros(target_size,target_size)

test_data_batch = test_data.unsqueeze(1)

logits_test = m(test_data_batch)

logits_test = F.softmax(logits_test,dim = 1)

for logs in range(len(logits_test)):
    row = torch.multinomial(logits_test[logs],1)
    confusion[row,test_targets[logs]] +=1
print(confusion)






