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
from progress.bar import IncrementalBar
from progress.colors import bold

point_0 = 2.53e1
point_0_un = 2e-2


#%%
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
                    func.remove_circle(px, py, x, y, r, test_mask)
                    # print('masked the image')
                  
            data_test *= test_mask
            func.see_image(data_test)
            
        elif max_val < edge:
            print('too faint')
            break # don't want things fainter than background
        # elif max_val < local_edge:
        #     print('too faint')
        #     continue
#%%

test_data = copy.deepcopy(obj_final) 

# create a masking array
test_mask = np.ones(np.shape(obj_final))

test_counter, test_source_flux = func.detect_sources(test_data, test_mask)

func.see_image(test_data)
#%
# =============================================================================
# 5.5 Calibrating the fluxes. 
# Converting instrumental counts to source magnitude
# =============================================================================

# converting counts into instrumental magnitude 
inst_mag = -2.5 * np.log10(test_source_flux)

# converting instrumental arguments into calibrated magnitudes 
mag = point_0 + inst_mag # point_0 defined at the top 
# for point in mag:
#     if mag >= 13:
          
# uncertainty in magnitude 
mag_fit, mag_cov, mag_centers, mag_freq = func.histogram_fit(mag, nbins=20, fit_func=func.exponential, p0=[1,1.1,8.4], plot=True, xlim1=0, xlim2=15, log_plot=True)
# plt.hist(mag,bins=15)
plt.show()

# =============================================================================
# 6.1 Comparison to liner fit
# =============================================================================

# determining the numbner of sources detected brighter than a magntitude limit 
limit = 3631e-26 # AB system magntiutde
pixl_deg = np.pi / (0.258 * 648000) # conversion factor from pixl to deg 

area = np.shape(test_data)[0] * np.shape(test_data)[1] * pixl_deg * pixl_deg # finding area of whole image

##### PLOTTING THE LOGARITHM ######
N = [] # number of sources above a given magnitude 
for mag_cutoff in mag_centers: # take the histogram points to be the cut off
    number = 0
    for each_mag in mag:
        if each_mag < mag_cutoff:
            number += 1
    N.append(number)

N_norm = np.array(N)/area # changing to number per degree
log_N = np.log10(N_norm)

# plot the data
log_fit, log_cov = func.plot_with_best_fit(
    x = mag_centers, 
    y = log_N, 
    title = '', 
    data_label = 'Collected Data', 
    fit_label = 'Linear Fit', 
    x_label = 'magnitude', 
    y_label = 'log$_{10}$(<m)', 
    data_colour = 'red', 
    fit_colour = 'black', 
    fit_func = func.linear)
plt.plot(mag_centers, log_N, 'x')

print(f'gradient ={log_fit[0]:.3f}')    
     
#%% Detecting non-circular objects
test_counter = 0
edge = mu_noise + 4 * sigma_noise
           
test_data = copy.deepcopy(obj_final) 

# create a masking array
test_mask = np.ones(np.shape(obj_final))

func.see_image(test_data)

test_source_flux = func.detect_sources(test_data, test_mask, test_counter)

func.see_image(test_data)
#            
# converting counts into instrumental magnitude 
inst_mag = -2.5 * np.log10(test_source_flux)

# converting instrumental arguments into calibrated magnitudes 
mag = point_0 + inst_mag # point_0 defined at the top 
# for point in mag:
#     if mag >= 13:
          
# uncertainty in magnitude 
mag_centers, mag_freq = func.histogram_fit(mag, nbins=20, fit_func=func.exponential, p0=[1,1.1,8.4], plot=True, xlim1=0, xlim2=15, log_plot=True)
# plt.hist(mag,bins=15)
plt.show()

# =============================================================================
# 6.1 Comparison to liner fit
# =============================================================================

# determining the numbner of sources detected brighter than a magntitude limit 
limit = 3631e-26 # AB system magntiutde
pixl_deg = np.pi / (0.258 * 648000) # conversion factor from pixl to deg 

area = np.shape(test_data)[0] * np.shape(test_data)[1] * pixl_deg * pixl_deg # finding area of whole image

##### PLOTTING THE LOGARITHM ######
N = [] # number of sources above a given magnitude 
for mag_cutoff in mag_centers: # take the histogram points to be the cut off
    number = 0
    for each_mag in mag:
        if each_mag < mag_cutoff:
            number += 1
    N.append(number)

N_norm = np.array(N)/area # changing to number per degree
log_N = np.log10(N_norm)

# plot the data
log_fit, log_cov = func.plot_with_best_fit(
    x = mag_centers, 
    y = log_N, 
    title = '', 
    data_label = 'Collected Data', 
    fit_label = 'Linear Fit', 
    x_label = 'magnitude', 
    y_label = 'log$_{10}$(<m)', 
    data_colour = 'red', 
    fit_colour = 'black', 
    fit_func = func.linear)
plt.plot(mag_centers, log_N, 'x')

print(f'gradient ={log_fit[0]:.3f}')              
            
            
                    
