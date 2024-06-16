# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:01:56 2024

A recreation of the MLPName.py but with native pytorch modules.

@author: James
"""
import torch
import torch.nn.functional as F


#Load Data ---------------------------------------------------
file = open("names.txt")
words = []

line = file.readline()
while line:
    words.append(line[:-1])
    line = file.readline()

#Model Vocab ------------------------------------------------
chars = sorted(set("".join(words)))
stoi = dict()
itos = dict()
for ch,i in zip(chars,range(len(chars))):
    stoi[ch] = i
    itos[i] = ch
stoi['.'] = len(stoi)
itos[len(itos)] = '.'
# print(stoi)
# print(itos)

#Data->Vocab conversion------------------------------------------
blockSize = 4
xinputs = []
yinputs = []
for word in words:
    word = "."*blockSize + word + "."
        
    for i in range(len(word[:-(blockSize)])):
        block = []
        for j in range(blockSize):
            block.append(stoi[word[i+j]])
        xinputs.append(block)
        yinputs.append(stoi[word[i+blockSize]])
        
xinputs = torch.tensor(xinputs)
yinputs = torch.tensor(yinputs)
    
#model initialisation--------------------------------------------------
encDim = 3
neuronPLayer = 50
# C = torch.randn((len(stoi),encDim))

layers = [  torch.nn.Embedding(len(stoi), encDim),torch.nn.Flatten(1,2),
            torch.nn.Linear(blockSize*encDim,neuronPLayer,dtype=torch.float,bias = False), torch.nn.BatchNorm1d(neuronPLayer,momentum=0.01) ,torch.nn.Tanh(),
            torch.nn.Linear(neuronPLayer,len(chars)+1,dtype=torch.float,bias = False),
          ]

parameters = [p for layer in layers for p in layer.parameters()]# + [C]

with torch.no_grad():
    #make last layer unconfident which minimises initial loss
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer,torch.nn.Linear):
            layer.weight *=5/3 #Ideal width for dist. entering tanh


for p in parameters:
    p.requires_grad = True


#Model Training-------------------------------------------------------
batchSize = 75
lr = 0.01
for i in range(150000):
    
    if i == 115000:
        lr*=0.1
    
    #separate out batch
    batch = torch.randint(0,len(xinputs),(batchSize,))
    xi = xinputs[batch]
    yi = yinputs[batch]

    x = xi
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x,yi)        
    if i%1000 == 0:
        print("Step: {}".format(i))
    if i%10000 ==0:
        print("Loss: {}".format(loss.item()))
    
    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    #parameter adjustment
    for p in parameters:
        p.data += -lr*p.grad
    

#Sampling the network------------------------------------------------
for layer in layers:
    layer.eval()

for _ in range(50):
    ch = ''
    word = [stoi['.'] for _ in range(blockSize)]
    while ch != '.':
        x = torch.tensor(word[-blockSize:])
        x = x.unsqueeze(0)
        for layer in layers:
            x = layer(x)
        logits = F.softmax(x,dim=1)
        ch = torch.multinomial(logits,1).item()
        word.append(ch)       
        ch = itos[ch]
    print(''.join(itos[i] for i in word[blockSize:-1]))
    
