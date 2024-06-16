# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:40:39 2024

Foundations of info storage in pytorch. the values of gradients and nodes in
the framework are stored as "Value" objects. Operations involving these object
create objects of the same type (Value). Value objects store the operation used
to create them through a prototype function that is called during ._backward().

._backward() will use the appropriate rules of differentiation to distribute
its gradient to the parents of the value. In this way, networks of value 
objects can backpropogate loss gradients and GD can be performed.

MLPs can be created by stacking lists of these Value objects

@author: James
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random



class Value:
    
    def __init__(self,data,_children = (),_op = '',label = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op  = _op
        self.label = label
        self._backward = lambda : None
             
    def __repr__(self):
        return "Value(label = {};data = {},grad = {})".format(self.label,self.data,self.grad)
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __radd__(self,other):
        return self+other
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        
        return self + (-other)
    
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only ints or floats"
        out = Value(self.data**other,(self,),'**{}'.format(other))
        
        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),label = 'exp')
        
        def _backward():
            self.grad += out.data*self.grad
        out._backward = _backward
        
    def __rmul__(self,other):
        return  self*other
    
    def __truediv__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        return self*other**-1

    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad = (1-t**2) * out.grad
        out._backward = _backward
        return out
    
    def __print__(self):
        return "Value(label = {};data = {})".format(self.label,self.data)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
 
        
class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)) , self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self,nin,nouts):
        #nouts: sizes of the layers
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
            # print(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
 
n = MLP(3,[4,4,1])

xs = [
      [2.0,3.0,-1.0],
      [3.0,-1.0,0.5],
      [0.5,1.0,1.0],
      [1.0,1.0,-1.0],
      ]
ys = [1.0,-1.0,-1.0,1.0]    
 
for k in range(1000):
    
    #Predict target class
    ypred = [n(x) for x in xs]
    
    #Calculate loss
    loss = sum((yout-ygt)**2 for yout,ygt in zip(ypred,ys))
   
    #Propogate loss (and reset grad)
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    #Fetch parameters and adjust against the gradient
    stepSize = 0.01
    for p in n.parameters():
        p.data += -1*stepSize*p.grad
    if k%100 == 0:
        print("Step: {}; Loss: {:.5f}".format(k,loss.data))