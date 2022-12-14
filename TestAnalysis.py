#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:20:10 2022

@author: eviebrass
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
import CatalogueSaving as s
import FunctionsFile as func
import csv
from Test import obj_final, mu_noise, sigma_noise
import FunctionsFile as func
import copy

edge = mu_noise + 2 * sigma_noise

# create a masking array
test_mask = np.ones(np.shape(obj_final))
# create a test counter
test_counter = 0 

data_test = copy.deepcopy(obj_final)

for i in range(0, 20000): # testing by fixing the number of sources we want to count
    if i % 1 == 0:    
        print(f'{i=}') # only want to print multiples of 100
    max_val, ylocs, xlocs = func.find_source(data_test)
    print(f'{max_val=}, {xlocs=}, {ylocs=}')
    for x, y in zip(xlocs, ylocs):
        # print(f'{x=},{y=}')
        r, local_edge = func.source_radius(data_test, x, y, nbins=10, p0=[8e4, mu_noise, sigma_noise], plot=False, xlim1=-5, xlim2=30)
        print(f'{r=}, {local_edge=}')
        if max_val > 0: 
            if r <= 3: # not counting small objects
                test_mask[y,x] = 0 # remove 1 random bright pixel
                data_test *= test_mask
                print('found small object')
                continue
            
            test_counter += 1
        
            for px in range(100):
                for py in range(100):
                    func.remove_circle(px, py, x, y, r, test_mask) # make sure flip x and y here
                    # print('masked the image')
                  
            data_test *= test_mask
            func.see_image(data_test)
            
        elif max_val < edge:
            print('too faint')
            break # don't want things fainter than background
        # elif max_val < local_edge:
        #     print('too faint')
        #     continue

            
            
            
            
            
            
            
                    
