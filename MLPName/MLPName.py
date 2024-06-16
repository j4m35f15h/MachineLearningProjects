# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:27:56 2024

MLP for name generation. Reads in a context size of 4 to predict the next
character.

@author: James
"""


import numpy as np
import torch
import torch.nn.functional as F

#Obtain raw data------------------------------------------------------
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
blockSize = 4

X = []
Y = []
for word in words:
    init = [26]*blockSize
    for ch in word+'.':
        newVal = stoi[ch]
        X.append(init)
        Y.append(newVal)
        init = init[1:]+[newVal]
X = torch.tensor(X)
Y = torch.tensor(Y)

#Model Setup---------------------------------------------------------
#2 Layer MLP
encDim = 3
NeuronPLayer = 50

C = torch.randn((len(chars)+1,encDim))

w1 = torch.randn((blockSize*encDim,NeuronPLayer))
b1 = torch.randn(NeuronPLayer)

w2 = torch.randn((NeuronPLayer,len(chars)+1))
b2 = torch.randn(len(chars)+1)

parameters = [C,w1,b1,w2,b2]

#Make gradients available for the coefficients
for p in parameters:
    p.requires_grad = True

#Model Training---------------------------------------------------
stepSize = 0.01
for i in range(200000):
    if i%1000 == 0:
        print(i)
        
    #forward pass
    batch = torch.randint(0,len(words),(30,))
    enc = C[X[batch]].view(-1,blockSize*encDim) #Add batching later
    l1 = torch.tanh(enc@w1 + b1)
    
    l2 = l1@w2 + b2
    loss = F.cross_entropy(l2,Y[batch])

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    #Adjust parameters
    for p in parameters:
        p.data += -stepSize*p.grad



#Sampling our network-----------------------------------------------
newWords = []
newWord = '.'
init = [stoi['.'] for _ in range(blockSize)]
for _ in range(20):
    ch = 'a'
    word = [stoi['.'] for _ in range(blockSize)]
    while ch != '.':
        enc = C[word[-blockSize:]].view(-1,blockSize*encDim)
        l1 = torch.tanh(enc@w1 + b1)
        
        l2 =  l1@w2 + b2
        probOut = F.softmax(l2,dim = 1)  
        
        chi = torch.multinomial(probOut,num_samples=1).item()
        ch = itos[chi]
        word.append(chi)
    
    print(''.join(itos[i] for i in word[blockSize:-1]))