#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:12 2022

@author: eviebrass
"""

# Code for testing 
import numpy as np
import matplotlib.pyplot as plt
import random
from astropy.io import fits
import FunctionsFile as func

# defining simple data sets for testing
empty_test = np.zeros((4172,2135)) # empty
same_val_test = np.ones((4172,2135)) # same value
noise_test = np.random.rand(4172,2135) # noise

# defining a function that outputs 2d gaussian blobs
def gauss_2d(size, x_center, y_center, sigma, A):
    ''' size gives the size of the whole image'''
    x = np.arange(0, size) # obtaing all the integer x values in the image
    y = np.arange(0, size) # obtaing all the integer y values in the image
    xx, yy = np.meshgrid(x, y) # creates 2d matrix of the two x and y arrays
    xx = xx - x_center # move the gaussian to specfific location
    yy = yy - y_center
    sigma = 9.1085 # used 1 arsec FWHM, convert to pixel then standard deviations
    power = (xx * xx) + (yy * yy) # index of the exponential in a guassian
    denom = 1/(2*np.pi*sigma*sigma)
    return A * denom * np.exp(-power/(2*sigma*sigma))  

def elipse(size, x_center ,y_center, sigma_x, sigma_y, theta, A):
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    xx_new = xx - x_center
    yy_new = yy - y_center
    cos = np.cos(theta)
    sin = np.sin(theta)
    a = (cos*cos) / (2*(sigma_x**2)) + (sin*sin) / (2*(sigma_y**2))
    b = -np.sin(2*theta) / (4*(sigma_x**2)) + np.sin(2*theta) / (4*(sigma_y**2))
    c = (sin*sin) / (2*(sigma_x**2)) + (cos*cos) / (2*(sigma_y**2))
    power = a*xx_new*xx_new + 2*b*xx_new*yy_new + c*yy_new*yy_new
    return A * np.exp(-power)

n_objects = 1 # number of objects -1

# generating random radii array
size = 200

random.seed(18)
loc_ran = [[random.randint(0, size) for i in range(n_objects)] for j in range(2)] # creating the centers of the code
test_obj = []
test_elip = [] #2d ellips

for i in range(n_objects): # same as i value
    x_current = loc_ran[0][i]
    y_current = loc_ran[1][i]
    obj_n = gauss_2d(size, x_current, y_current, sigma=1, A=100000)
    test_obj.append(obj_n)

# standard sigma size 
sigma =  (1/0.258)*2.35 # used 1 arsec FWHM, convert to pixel then sd

# generating random angle array for elipse 
theta_rand = [random.uniform(0,np.pi) for i in range(n_objects)]

# creating roated elipses 
for i in range(n_objects): # same as i value
    x_current = loc_ran[0][i]
    y_current = loc_ran[1][i]
    theta_current = theta_rand[i]
    elip_n = elipse(size, x_current , y_current , sigma, sigma/2, theta_current , 1000)
    test_elip.append(elip_n)

# obj = sum(test_obj) # creating an object 
obj = sum(test_elip)

# add noise to the background of the test
mu_noise, sigma_noise = 3420, 18
noise = np.random.normal(mu_noise, sigma_noise, size=(size,size))
obj_final = obj + noise

func.see_image(obj_final)

# fits.writeto('test.fits', obj_final, overwrite=True)
