# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:18:55 2024

VAE acting as a denoiser. 

Uses a generated data set, contining 5000 samples 
consisting of 100 dimensions. Each dimension is provided a value equal to the
number of the dimension added to a normally distributed random variable 
multiplied by the number of the dimension. The VAE extracts the common value
across the samples for each dimension (i.e. the number of the dimension).

3 visualisations are used to assess the error

@author: James
"""

import torch
import torch.nn as nn
from torch.nn import functional as F 

import matplotlib.pyplot as plt

#Hyper paramter set up -----------------------------------------------
device = 'cude' if torch.cuda.is_available() else 'cpu'
data_size = 100
encoder_layer_size = [300,50]
decoder_layer_size = [50,100]
latent_size = 10

dropout = 0.25

training_rounds = 3000
learning_rate = 3e-4
eval_it = 100 #Signposting during training
eval_counts = 100 #Sample count assessed during signposting
batch_size = 30

#Data input------------------------------------------------------------

print("Generating Data...")

data = torch.zeros(5000,100,1)
S,G,R = data.shape
for s in range(S):
    for g in range(G):
        data[s,g,0] = g+1 + torch.randn((1,))*(g+1)

print("Complete")
    
pre_norm_data = torch.tensor(data)
data_mean = torch.mean(data,dim = 0)
data_std = torch.std(data,dim = 0)
data = (data-data_mean)/data_std      #model data has 0 mean & unit variance
#testTrain Split ----------------------------------------


n = int(0.9*S)
train_data = data[:n,:,:].view(-1,data_size)
test_data  = data[n:,:,:].view(-1,data_size)

#Utility Functions------------------------------------------------------
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    
    ix = torch.randint(data.shape[0],(batch_size,))
    x = torch.stack([data[i,:] for i in ix])
    y = torch.stack([data[i,:] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y


@torch.no_grad
def evaluate_loss():
    #Combined MSE and KL Divergence is used for loss
    out = {}
    m.eval()
    
    for split in ['train','test']:
        losses = torch.zeros(eval_counts)
        
        for i in range(eval_counts):
            X,Y = get_batch(split)
            pred,mu,std = m(X)
            loss = F.mse_loss(pred,Y) - 0.5*torch.sum(1. + torch.log(std**2) - std**2 - mu**2)
            loss = torch.sum(loss)
            losses[i] = loss.item()
        
        out[split] = losses.mean()
        
    m.train()
    
    return out


#Layer definitions------------------------------------------------------

class VAE_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = [
            nn.Linear(data_size,encoder_layer_size[0]),
            nn.BatchNorm1d(encoder_layer_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
            ]
        
        if len(encoder_layer_size) > 1:
            for i in range(1,len(encoder_layer_size)):
                self.encoder += [
                    nn.Linear(encoder_layer_size[i-1],encoder_layer_size[i]),
                    nn.BatchNorm1d(encoder_layer_size[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                    ]
        
        self.encoder = nn.Sequential(*self.encoder)
        
    def forward(self,x):
        x = self.encoder(x)
        return x

class VAE_latent(nn.Module):
    def __init__(self):
        super().__init__()
        self.Mu_latent = nn.Linear(encoder_layer_size[-1],latent_size)
        self.Var_latent = nn.Linear(encoder_layer_size[-1],latent_size)
    
    def forward(self,x):
        mu_latent = self.Mu_latent(x)
        
        var_latent = self.Var_latent(x)
        std_latent = torch.exp(var_latent*0.5)
        
        z = mu_latent + std_latent * torch.rand_like(std_latent)
        
        return mu_latent,std_latent,z

class VAE_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = [
            nn.Linear(latent_size,decoder_layer_size[0]),
            nn.BatchNorm1d(decoder_layer_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
            ]
        
        if len(decoder_layer_size) > 1:
            for i in range(1,len(decoder_layer_size)):
                self.decoder+= [
                    nn.Linear(decoder_layer_size[i-1],decoder_layer_size[i]),
                    nn.BatchNorm1d(decoder_layer_size[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                    ]
        
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self,x):
        x = self.decoder(x)
        return x
        

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_encoder()
        self.latent = VAE_latent()
        self.decoder = VAE_decoder()
        
    def forward(self,x):
        x = self.encoder(x)
        mu,std,z = self.latent(x)
        output = self.decoder(z)
        
        return output,mu,std
       
        
       
#Model creation-------------------------------------------------------
m = VAE()
m.to(device)
optimiser = torch.optim.Adam(m.parameters(),lr = learning_rate)

#Training loop--------------------------------------------------------
print("Training model")

for i in range(training_rounds):
    
    xb,yb = get_batch('train')

    y_pred,mu,std = m(xb)

    loss = F.mse_loss(y_pred,yb) - 0.5* torch.sum(1. + torch.log(std**2) - std**2 - mu**2)

    optimiser.zero_grad(set_to_none = True)
    loss.backward() 
    optimiser.step()
    
    if i % eval_it == 0:
        losses = evaluate_loss()
        print("Step {}:   train loss = {:.5f}, test loss = {:.5f}".format(i,losses['train'],losses['test']))

#Visualisation of a sample-------------------------------------------
"""
The data produced is calculated by adding a normally distributed value to
the line y=x. Three graphs were produced to visualise the results; the first 
plots a prediction alongised y=x, the second plots the difference between
the two lines (i.e. error), and the third plots the % error
"""
with torch.no_grad():
    test_pred,_,_ = m(data[1,:].view(1,-1))
    test_pred = test_pred*data_std.view(1,-1) + data_mean.view(1,-1)
    
    plt.figure()
    plt.title("Prediction vs Mean")
    plt.xlabel("Dim")
    plt.ylabel("Value")
    plt.plot(range(int(test_pred.shape[1])),test_pred[0,:])
    plt.plot(range(int(test_pred.shape[1])),range(1,int(test_pred.shape[1]+1)))
    
    plt.figure()
    plt.title("Prediction error")
    plt.xlabel("Dim")
    plt.ylabel("Error")
    plt.plot(range(int(test_pred.shape[1])),test_pred[0,:]-torch.tensor([i+1 for i in range(100)]))
    plt.plot(range(int(test_pred.shape[1])),torch.zeros(test_pred.shape[1]))
    
    plt.figure()
    plt.title("Normalised Prediction error")
    plt.xlabel("Dim")
    plt.ylabel("% Error")
    plt.plot(range(int(test_pred.shape[1])),100*(test_pred[0,:]-torch.tensor([i+1 for i in range(100)]))/test_pred[0,:])
    plt.plot(range(int(test_pred.shape[1])),torch.zeros(test_pred.shape[1]))
    






