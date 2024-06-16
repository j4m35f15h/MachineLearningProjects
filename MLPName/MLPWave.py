# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:23:26 2024

MLP that takes inspitration from the wavenet model: 
    arxiv.org/pdf/1609.03499
Implements the Dilated Casual Convolution whereby neurons in a layer 
communicate with adjacent pairs in the layer below. This reduces the rate of
information compression in the network, which may be lossy.

@author: James
"""

import numpy as np
import torch
import torch.nn.functional as F

#Obtain raw data----------------------------------------------
file = open("names.txt")

words = []
line = file.readline()
while line:
    words.append(line[:-1])
    line = file.readline()
file.close()

#Model Vocab ------------------------------------------------
chars = sorted(set(''.join(words)))
stoi = dict()
itos = dict()
for place,ch in enumerate(chars):
    itos.update({place:ch})
    stoi.update({ch:place})
stoi.update({'.':place+1})
itos.update({place+1:'.'})
    
#Block inputs for context-----------------------------------------
blockSize = 8

X = []
Y = []
for word in words:
    init = [26]*blockSize
    for ch in word+'.':
        newVal = stoi[ch]
        X.append(init)
        Y.append(newVal)
        #print(''.join(itos[i]for i in init),'----->',ch)
        init = init[1:]+[newVal]
X = torch.tensor(X)
Y = torch.tensor(Y)

#BatchNorm wierdness: Batchnorm want dim 1 to be the data, whereas in the 
#format here, dim 0,1 are batch dimensions while dim 2 is the data.
#Two things can be done, create a new layer to reshape the data, or create
#a homebrew batchnorm. Here I opt for the latter for practice:
    
class BatchNorm1d(torch.nn.Module):
    def __init__(self,dim, eps = 1e-5, momentum = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        #
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def forward(self,x):
        if self.training:
            #identify batch dimensions
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim,keepdim = True)
            xvar  = x.var(dim ,keepdim = True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum*xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum*xvar
        return self.out
                    
#Model Setup---------------------------------------------------------
encDim = 10
NeuronPLayer = 75

model = torch.nn.Sequential(torch.nn.Embedding(len(chars)+1,encDim),
                              torch.nn.Flatten(1,2),torch.nn.Unflatten(1, [blockSize//2,encDim*2]),
                              torch.nn.Linear(encDim*2,NeuronPLayer,dtype=torch.float,bias = False), 
                              BatchNorm1d(NeuronPLayer,momentum=0.01) ,torch.nn.Tanh(),
                                torch.nn.Flatten(1,2),torch.nn.Unflatten(1, [-1,NeuronPLayer*2]),
                                torch.nn.Linear(NeuronPLayer*2,NeuronPLayer,dtype=torch.float,bias = False), 
                                BatchNorm1d(NeuronPLayer,momentum=0.01) ,torch.nn.Tanh(),
                                    torch.nn.Flatten(1,2),torch.nn.Unflatten(1, [-1,NeuronPLayer*2]),
                                    torch.nn.Linear(NeuronPLayer*2,NeuronPLayer,dtype=torch.float,bias = False), 
                                    BatchNorm1d(NeuronPLayer,momentum=0.01) ,torch.nn.Tanh(),
                              torch.nn.Linear(NeuronPLayer,len(chars)+1,dtype=torch.float,bias = False),
                              )



with torch.no_grad():
    #make last layer unconfident which minimises initial loss
    model[-1].weight *= 0.1
    for layer in model[:-1]:
        if isinstance(layer,torch.nn.Linear):
            layer.weight *=5/3 #Ideal width for normal entering tanh

parameters = [p for p in model.parameters()]
for p in parameters:
    p.requires_grad = True

#Model Training---------------------------------------------------
stepSize = 0.01
for i in range(50000):
    if i == 75000:
        stepSize*=0.1
    
    #forward pass
    batch = torch.randint(0,len(words),(30,))
    logits = X[batch]
    
    logits = model(logits)
    
    loss = F.cross_entropy(logits.float().squeeze(1),Y[batch])
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    for p in parameters:
        p.data += -stepSize*p.grad
    
    if i%1000 == 0:
        print(i,loss.item())
    # break
    
##Sampling our network------------------------------------------

#Batchnorm needs eval call
for layer in model:
    layer.eval()


for _ in range(50):
    ch = ''
    word = [stoi['.'] for _ in range(blockSize)]
    while ch != '.':
        x = torch.tensor([word[-blockSize:]]) 
        #x = C[x]
        logits = model(x)
        logits = logits.float().squeeze(1)
        logits = F.softmax(logits,dim=1)
        ch = torch.multinomial(logits,1).item()
        word.append(ch)       
        ch = itos[ch]
    print(''.join(itos[i] for i in word[blockSize:-1]))

for layer in model:
    layer.train()
