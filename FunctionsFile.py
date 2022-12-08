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
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.show()

###### HISTOGRAM WITH GAUSSIAN FIT ######
def gaussian(data, A, mean, sigma):
    return A*np.exp(-1*(data - mean)**2 / (2*sigma*sigma))
    
def histogram_fit(data, nbins, title='', fit_func=gaussian, p0=[7e6,3420,18], plot=False):
    data_flat = np.ravel(data)
    hist_y, hist_edges = np.histogram(data, bins=nbins)
    hist_centers = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_error = np.sqrt(hist_y)
    hist_fit, hist_cov = curve_fit(gaussian, hist_centers, hist_y, p0)
    if plot == True:
        x_hist_fit = np.linspace(3300, 3650, 1000)
        plt.plot(x_hist_fit, fit_func(x_hist_fit, *hist_fit), color='black', label = 'gaussian fit')
        plt.errorbar(hist_centers, hist_y, yerr= hist_error, color='red', fmt='x')
        plt.hist(data_flat, bins=nbins, label ='Pixel Counts')
        plt.xlim(3300,3650)
        plt.title(title)
        plt.legend()
        plt.show()
    return hist_fit, hist_cov

###### IMAGE MASKING ######
def remove_circle(x_val, y_val, x_center, y_center, r, mask_array, photometry=0, pixl=0):
    l = r + 49
    reduced_mask = mask_array[y_center-l:y_center+l, x_center-l:x_center+l] # limiting the area to loop through
    y = x_val + x_center - l # the pixel number in the context of image
    x = y_val + y_center - l 
    x_mag = (x - y_center) * (x - y_center)
    y_mag = (y - x_center) * (y - x_center)
    r_sq = r * r
    
    # counter to determine the number of pixels associated with source 
    pixl_count = 0 
    
    if x_mag + y_mag < r_sq:
        pixl_count += 1 
        if photometry == 0:
            reduced_mask[x_val, y_val] = 0
        if photometry == 1:
            return True
    mask_array[y_center-l:y_center+l, x_center-l:x_center+l] = reduced_mask
    
    if pixl == 1:
        return pixl_count
        

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
def find_source(data, nbins = 4000, plot=False):
    max_val = np.max(data)
    locx, locy = np.where(data == max_val)
    locx = int(locx)
    locy = int(locy)
    r = 0 
    # take an area for the background calculations
    l = 50 # length to go either side of the source
    data_local = data[locx-l:locx+l, locy-l:locy+l] # picking out a square around the source
    background_fit, background_cov = histogram_fit(data_local, nbins, plot=plot)
    background = background_fit[1]
    sigma = background_fit[2]
    edge = background + 3 * sigma # anything below this is defined as background
    # print(f'{background=}')
    # print(f'{sigma=}')
    # print(f'{edge=}')
    # find the radius of the detected star
    data_scan = data[locx:locx+100, locy] # limit the region that we are searching
    # data_scan= data[locx, locy:locy+100]
    for x in range(len(data_scan)): # pick out the points along y to find radius
        if data_scan[x] < edge:
            r = x
            break
    return locx, locy, r, max_val, edge
    
















    