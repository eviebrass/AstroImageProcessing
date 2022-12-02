#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:43:13 2022

@author: ioanabalabasciuc
"""
# Code for testing 
import numpy as np
import matplotlib.pyplot as plt
import random

#defining simple data sets for testing
empty_test = np.zeros((4172,2135)) #empty
same_val_test = np.ones((4172,2135)) #same value
noise_test = np.random.rand(4172,2135) #noise

#defining a function that outputs gaussain blob to simular Gaussian blobs
def gauss_2d(size,x_offset,y_offset,sigma):
    x = np.arange(0,size)
    y = np.arange(0,size)
    xx,yy= np.meshgrid(x,y)
    xx= xx-x_offset
    yy= yy -y_offset
    sigma = 0.6063   #used 1 arsec FWHM, convert to pixel then sd
    power = (xx*xx) +(yy*yy)
    denom = 1/(2*np.pi*sigma*sigma)
    return denom* np.exp(-power/(2*sigma*sigma))  

n_objects = 10 #number of objects  -1

#generating random radii array
random.seed(20)
radi_ran = [[random.randint(0,100) for i in range(n_objects)] for j in range(2)]
test_obj = []

for i in range(n_objects): # same as i value
    obj_n = gauss_2d(100,radi_ran[0][i],radi_ran[1][i],1)
    test_obj.append(obj_n)
    
obj_final= sum(test_obj)
plt.imshow(obj_final,cmap='gray')


