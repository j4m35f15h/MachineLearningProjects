# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:14:01 2024

Not a very useful project: I recreated Linear regression with a neuron.
Added a visualisation function to show how the linear relationship that the
neuron predicts evolves with each training iteration.

@author: James
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Initialise the parameters for the data (the model will estimate these)
dataGrad,dataBias = torch.randn((1)),torch.randn((1))

#Generate datasets; Different Additions of noise were tested
dataSize = 100
xinput = torch.randn((dataSize))
yinput = xinput*dataGrad + dataBias
yinputV2 = xinput*dataGrad + dataBias + torch.randn((xinput.shape))*(0.1*(max(xinput)-min(xinput)))
yinputV3 = xinput*(dataGrad+torch.randn((xinput.shape))*0.15) + (dataBias+torch.randn((xinput.shape))*0.15)


#Set up for visualisation data
lineGrads = []
lineBias = []

#Model creation--------------------------------------------------
model = torch.nn.Linear(1,1,dtype=torch.float)


parameters = [p for p in model.parameters()]
for p in parameters:
    p.grad = None


#Hyper-parameters-------------------------------------------------
lr = 0.1
batchSize = dataSize//10
batchi = torch.randint(0,len(xinput)-1,xinput.size())
totalIt = 20

#Model training------------------------------------------------------
for i in range(totalIt):   

    #forward pass
    batch = torch.randint(0,len(xinput),(batchSize,))
    ytest = model(xinput[batch].view(batchSize,1))
    
    #Loss calc: mean squared error
    loss = sum((ytest - yinputV3[batch].view(-1,1))**2)/batchSize
    
    # if i%1 == 0:
    #     print(loss.item())
    
    #Store intermediate steps for later plotting
    if i % 2 == 0:
        lineGrads.append(torch.clone(model.weight))
        lineBias.append(torch.clone(model.bias))
        
    
    #backpass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    
    #parameter adjust
    for p in parameters:
        p.data += -p.grad*lr
    
#Plotting model progression---------------------------------------------

plt.scatter(xinput,yinputV3)
plot_step = (max(xinput)-min(xinput))/10
xrange = torch.arange(min(xinput)-plot_step,max(xinput)+plot_step,plot_step)

yrange = xrange*dataGrad + dataBias
plt.plot(xrange,yrange)
yLines = torch.tensor(lineGrads).view(-1,1)*xrange.view(1,-1) + torch.tensor(lineBias).view(totalIt//2,-1)
for i in range(yLines.shape[0]):
    plt.plot(xrange,yLines[i,:])
    
    #Intermediate grads
    # print(f"{lineGrads[i].item():0.5f}")

#Report difference in model and data-------------------------------------
modelGrad,modelBias = model.weight.item(),model.bias.item()
print("Data Grad = {:.5f} | Grad Estimate = {:.5f}".format(dataGrad.item(),modelGrad))
print("Data Bias = {:.5f} | Bias Estimate = {:.5f}".format(dataBias.item(),modelBias))
