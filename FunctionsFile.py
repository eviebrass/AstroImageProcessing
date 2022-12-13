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
    l = r + 10
    x = x_val + x_center - l # the pixel number in the context of image
    y = y_val + y_center - l 
    x_mag = (x - x_center) * (x - x_center)
    y_mag = (y - y_center) * (y - y_center)
    r_sq = r * r
    edge_case = 0 # if at an edge then this changes to 1
    # x_len = 4172
    # y_len = 2135
    x_len = 500
    y_len = 500
    
    # counter to determine the number of pixels associated with source 
    pixl_count = 0 
    
    if pixl == 1:
        return pixl_count
    ### dealing with sources close to the edges
    # set the mask array to not go past the edges
    if x_center+l > x_len:
        reduced_mask = mask_array[y_center-l:y_len, y_center-l:y_center+l]
        edge_case = 1
        # print('HIT RIGHT BOUNDARY')
    elif y_center+l > y_len:
        reduced_mask = mask_array[x_center-l:x_center+l, y_center-l:y_len]
        edge_case = 2
        # print('HIT TOP BOUNDARY')
    elif x_center-l < 0:
        reduced_mask = mask_array[0:x_center+l, y_center-l:y_center+l]
        edge_case = 3
        # print('HIT LEFT BOUNDARY')
    elif y_center-l < 0:
        reduced_mask = mask_array[x_center-l:x_center+l, 0:y_center+l]
        edge_case = 4
        # print('HIT BOTTOM BOUNDARY')
    else:
        reduced_mask = mask_array[x_center-l:x_center+l, y_center-l:y_center+l]
    
    xmax = np.shape(reduced_mask)[0] # dont want to go longer than the reduced mass size
    ymax = np.shape(reduced_mask)[1]
    
    if x_mag + y_mag < r_sq and x_val < xmax and y_val < ymax:
        # if not doing photometry then mask section
        if photometry == 0:
            reduced_mask[x_val, y_val] = 0
        if photometry == 1:
            return True
    
    # updating the mask array
    if edge_case == 0:
        mask_array[y_center-l:y_center+l, x_center-l:x_center+l] = reduced_mask
    elif edge_case == 1:
        mask_array[x_center-l:x_len, y_center-l:y_center+l] = reduced_mask
    elif edge_case == 2:
        mask_array[x_center-l:x_center+l, y_center-l:y_len] = reduced_mask
    elif edge_case == 3:
        mask_array[0:x_center+l, y_center-l:y_center+l] = reduced_mask
    elif edge_case == 4:
        mask_array[x_center-l:x_center+l, 0:y_center+l] = reduced_mask
         
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
    locx, locy = np.where(data == max_val)
    r = 0
    l = 50 # length to go either side of the source
    return max_val, locx, locy

def source_radius(data, locx, locy, nbins = 4000, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650):
    r = 0 
    locx = int(locx)
    locy = int(locy)
    # pick out an area around the source
    l = 20 
    data_local = data[locx-l:locx+l, locy-l:locy+l]
    background_fit, background_cov = histogram_fit(data_local, nbins, p0=p0, plot=plot, xlim1=xlim1, xlim2=xlim2)
    background = background_fit[1]
    sigma = background_fit[2]
    edge = background + 2 * sigma # anything below this is defined as background
    # find the radius of the detected star
    data_scan = data[locx:locx+1000, locy] # limit the region that we are searching
    for x in range(len(data_scan)): # pick out the points along y to find radius
        # print(f'{data_scan[x]=}')
        if data_scan[x] <= edge:
            r = x
            break
    return r, edge

















    