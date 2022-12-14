#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:14:11 2022

@author: eviebrass
"""
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

###### SEE THE IMAGE WE ARE LOOKING AT ######
def see_image(data):
    plt.imshow(data, cmap='gray', origin='lower')
    plt.colorbar()
    plt.show()

###### HISTOGRAM WITH GAUSSIAN FIT ######
def gaussian(data, A, mean, sigma):
    return A*np.exp(-1*(data - mean)**2 / (2*sigma*sigma))
    
def histogram_fit(data, nbins, title='', fit_func=gaussian, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650):
    data_flat = np.ravel(data)
    hist_y, hist_edges = np.histogram(data, bins=nbins)
    hist_centers = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_error = np.sqrt(hist_y)
    hist_fit, hist_cov = curve_fit(gaussian, hist_centers, hist_y, p0)
    if plot == True:
        x_hist_fit = np.linspace(xlim1, xlim2, 1000)
        plt.plot(x_hist_fit, fit_func(x_hist_fit, *hist_fit), color='black', label = 'gaussian fit')
        plt.errorbar(hist_centers, hist_y, yerr= hist_error, color='red', fmt='x')
        plt.hist(data_flat, bins=nbins, label ='Pixel Counts')
        plt.xlim(xlim1,xlim2)
        plt.title(title)
        plt.legend()
        plt.show()
    return hist_fit, hist_cov

###### IMAGE MASKING ######
def remove_circle(x_val, y_val, x_center, y_center, r, mask_array, photometry=0, pixl=0):
    # print(f'{x_center=}, {y_center=}')
    l = r + 50
    x = x_val + x_center - l # the pixel number in the context of image
    y = y_val + y_center - l
    y_len = 4172
    x_len = 2135
    # x_len = 500
    # y_len = 500
    
    # counter to determine the number of pixels associated with source 
    pixl_count = 0 
    
    if pixl == 1:
        return pixl_count
    
    ### dealing with sources close to the edges (including corners)
    if x_center+l > x_len:
        reduced_mask = mask_array[y_center-l:y_center+l+1, x_center-l:x_len+1]
        if y_center+l > y_len: 
            reduced_mask = mask_array[y_center-l:y_len+1, x_center-l:x_len+1]
            # print('HIT TOP RIGHT CORNER')
        if y_center-l < 0:
            y = y_val
            reduced_mask =  mask_array[0:y_center+l+1, x_center-l:x_len+1]
            # print('HIT BOTTOM RIGHT CORNER)
        # print('HIT RIGHT BOUNDARY')
    
    elif y_center+l > y_len:
        reduced_mask = mask_array[y_center-l:y_len+1, x_center-l:x_center+l+1]
        # print('HIT TOP BOUNDARY')
   
    elif x_center-l < 0:
        x = x_val
        reduced_mask = mask_array[y_center-l:y_center+l+1, 0:x_center+l+1]
        if y_center+l > y_len:
            reduced_mask = mask_array[y_center-l:y_len+1, 0:x_center+l+1]
            # print('HIT BOTTOM RIGHT CORNER')
        if y_center-l < 0:
            y = y_val
            reduced_mask = mask_array[0:y_center+l+1, 0:x_center+l+1]
            # print('HIT BOTTOM LEFT CORNER)
        # print('HIT LEFT BOUNDARY')
    
    elif y_center-l < 0:
        y = y_val
        reduced_mask = mask_array[0:y_center+l+1, x_center-l:x_center+l+1]
        # print('HIT BOTTOM BOUNDARY')
   
    else:
        reduced_mask = mask_array[y_center-l:y_center+l+1, x_center-l:x_center+l+1]
    
    
    x_mag = (x - x_center) * (x - x_center)
    y_mag = (y - y_center) * (y - y_center)
    r_sq = r * r
    xmax = np.shape(reduced_mask)[1] # dont want to go longer than the reduced mass size
    ymax = np.shape(reduced_mask)[0]
    
    if x_mag + y_mag < r_sq and x_val < xmax and y_val < ymax: 
        # xmax and ymax mean don't go outside mask size
        # if not doing photometry then mask section
        if photometry == 0:
            reduced_mask[y_val, x_val] = 0
        if photometry == 1:
            return True
         
def remove_triangle(x_val, y_val, x1, y1, x2, y2, x3, y3, mask_array):
    # gradients of each side of the triangle
    m1 = (x3 - x1) / (y3 - y1)
    m2 = (x3 - x2) / (y3 - y2)
    # constants for each straight line
    c1 = x1 - m1 * y1
    c2 = x2 - m2 * y2
    if y_val > m1*x_val + c1 and y_val < m2*x_val + c2 and x_val > y1:
        mask_array[x_val, y_val] = 0

def remove_rect(x1, x2, y1, y2, mask_array):
    mask_array[y1:y2, x1:x2] = 0
    
###### DETECTING SOURCES ######
def find_source(data):
    max_val = np.max(data)
    locy, locx = np.where(data == max_val)
    # flip x and y because how np.where works
    # matrix i=y, j=x
    r = 0
    l = 50 # length to go either side of the source
    return max_val, locy, locx

def source_radius(data, locx, locy, nbins = 4000, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650):
    r = 0 
    locx = int(locx)
    locy = int(locy)
    # pick out an area around the source
    l = 40 
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    data_local = data[locx-l:locx+l, locy-l:locy+l]
    background_fit, background_cov = histogram_fit(data_local, nbins, p0=p0, plot=plot, xlim1=xlim1, xlim2=xlim2)
    background = background_fit[1]
    sigma = background_fit[2]
    edge = background + 3 * sigma # anything below this is defined as background
   
    # find the radius of the detected star
    data_scan1 = data[locy:locy+30, locx] # limit the region that we are searching
    #swapped x and y around as matrix are defined the other way round
    for xR in range(len(data_scan1)): # pick out the points along y to find radius
        if data_scan1[xR] <= edge:
            r1 = xR
            break
    data_scan2 = data[locy-30:locy, locx]
    for xL in range(len(data_scan1)): # pick out the points along y to find radius
        if data_scan1[xL] <= edge:
            r2 = xL
            break
        
    data_scan3 = data[locy, locx:locx+30]
    for yU in range(len(data_scan3)):
        if data_scan3[yU] <= edge:
            r3 = yU
            break
    data_scan4 = data[locy, locx-30:locx]
    for yD in range(len(data_scan4)):
        if data_scan4[yD] <= edge:
            r4 = yD
            break  
        
    r = np.max([r1, r2, r3, r4])
    return r, edge

















    