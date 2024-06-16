# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:27:57 2024

@author: James
"""


from sklearn.datasets import load_digits

input = load_digits()

f = open("digits.txt","w")
for image in range(len(input.images)):
    line = ""
    for i in input.images[image]:
        for j in i:
            
            line+=str(j)+ " "
    f.write(line+"\n")
    
    f.write(str(input.target[image])+"\n")
f.close()

