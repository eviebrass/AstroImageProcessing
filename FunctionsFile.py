#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:14:11 2022

@author: eviebrass
"""
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
# imports for checking progress
import time as time
from progress.bar import IncrementalBar
from progress.colors import bold

###### SEE THE IMAGE WE ARE LOOKING AT ######
def see_image(data):
    plt.imshow(data, cmap='gray', origin='lower')
    plt.colorbar()
    plt.show()

###### FUNCTIONS FOR FITTING ######
def gaussian(data, A, mean, sigma):
    return A*np.exp(-1*(data - mean)**2 / (2*sigma*sigma))

def exponential(data, A, alpha,c):
    return A * np.exp(alpha*(data-c))

def linear(data, m, c):
    return m * data + c

###### PLOTTING DATA WITH A FIT ######
def plot_data(x, y, xerr, yerr, data_colour, label):
    with plt.style.context('ggplot'):
        plt.errorbar(x, y, yerr, xerr, 'x',color=data_colour, label=label)

def plot_best_fit( x, y, fit_func, weight, intial_guess,fit_colour, label):
    fit, cov = curve_fit(fit_func, x, y, p0=intial_guess, sigma = weight)
    with plt.style.context('ggplot'):
        plt.plot(x, fit_func(x, *fit), color=fit_colour, label=label)
    return fit, cov

def plot_with_best_fit(
    x,
    y,
    title,
    data_label,
    fit_label,
    x_label,
    y_label,
    data_colour,
    fit_colour,
    fit_func,
    weight = None, 
    initial_guess = None,
    xerr=0,
    yerr=0,
    beginfit=None,
    endfit=None,
    points=True,
):
    with plt.style.context('ggplot'):
        # fig, ax = plt.subplots()
        # if points == True:
        if np.any(weight != None):
            weight = weight[beginfit:endfit]
        if points == True:
            plot_data(x, y, xerr, yerr, data_colour, data_label)
        fit,cov = plot_best_fit(x[beginfit:endfit], y[beginfit:endfit], fit_func, weight, initial_guess, fit_colour, fit_label)
        # print(f'fit:{fit}')
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        # plt.show()
    return fit, cov

###### PLOTTING A HISTOGRAM WITH A FIT ######
def histogram_fit(data, nbins, title='', fit_func=gaussian, p0=[7e6,3400,18], plot=False, xlim1=3300, xlim2=3650, log_plot=False):
    '''
    Plotting histogram to determine gaussian fit of flux hist 
    nbins is the number of bins  
    fit_func is the gaussian function 
    plot shows image is true
    x_lims1,2 are x-axis range
    log_plot determines returns hist with uncertainty if true 
    '''
    # turn the data into a 1D array
    data_flat = np.ravel(data)
    # remove any low valued points that affect fitting from the array that are due to removed sources
    data_flat_clean = data_flat[data_flat > 0]
    
    if len(data_flat_clean) == 0: # return initial guess if only have zeros in data
        return p0, np.zeros((3,3))
    
    hist_y, hist_edges = np.histogram(data_flat_clean, bins=nbins)
    hist_centers = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_error = np.sqrt(hist_y)
    
    # remove the zero frequency points to fit the gaussian
    zeros = np.where(hist_y==0) # find the indices of zero points
    y_lst = list(hist_y) # keep only the non-zero points
    centers_lst = list(hist_centers)# turns centers into list 
    for index in sorted(zeros[0], reverse=True): # delete the zero points in reverse order
        del y_lst[index]    
        del centers_lst[index]
    hist_y_clean = np.array(y_lst)
    hist_centers_clean = np.array(centers_lst) # turn list back into array to fit
    hist_fit, hist_cov = curve_fit(fit_func, hist_centers_clean, hist_y_clean, p0)
    
    if plot == True:
        x_hist_fit = np.linspace(xlim1, xlim2, 1000)
        plt.plot(x_hist_fit, fit_func(x_hist_fit, *hist_fit), color='black', label = 'gaussian fit')
        plt.errorbar(hist_centers, hist_y, yerr= hist_error, color='red', fmt='x')
        plt.hist(data_flat_clean, bins=nbins, label ='Pixel Counts')
        # plt.xlim(xlim1,xlim2)
        plt.title(title)
        plt.legend()
        plt.show()
        
    if log_plot == False:
        return hist_fit, hist_cov
    
    elif log_plot == True:
        return hist_centers, hist_y

###### FINDING A BIT OF DATA THAT YOU WANT TO ANALYSE ######
def reduce_data(data, mask, x1, x2, y1, y2):
    data_reduced = data[y1:y2, x1:x2]
    mask_reduced = mask[y1:y2, x1:x2]
    return data_reduced, mask_reduced
    
###### IMAGE MASKING ######
def remove_circle(x_val, y_val, x_center, y_center, r, data, mask_array,y_len=4172, x_len=2135, photometry=0):
    ''' 
    Masking objects assuming circular shapes 
    x_val and y_val are all the coordiante values in image 
    (x_centre,y_centre) are centre coordinates for object 
    r is the radius of the objects  
    mask_array is the most up to date mask array 
    photometry obtains the flux of each pixel in aperture if true
    pixl counts the number of pixels in given aperture if true 
    '''
    # print(f'{x_center=}, {y_center=}')
    l = r + 20
    x = x_val + x_center - l # the pixel number in the context of image
    y = y_val + y_center - l
    
    # reduced mask for points not near the edges
    reduced_mask = mask_array[y_center-l:y_center+l+1, x_center-l:x_center+l+1]
    reduced_data = data[y_center-l:y_center+l+1, x_center-l:x_center+l+1]
    
    ### dealing with sources close to the edges (including corners)
    if x_center+l > x_len:
    # print('HIT RIGHT BOUNDARY')
        reduced_mask = mask_array[y_center-l:y_center+l+1, x_center-l:x_len+1]
        reduced_data = data[y_center-l:y_center+l+1, x_center-l:x_len+1]
        if y_center+l > y_len: 
        # print('HIT TOP RIGHT CORNER')
            reduced_mask = mask_array[y_center-l:y_len+1, x_center-l:x_len+1]
            reduced_data = data[y_center-l:y_len+1, x_center-l:x_len+1]
        if y_center-l < 0:
            # print('HIT BOTTOM RIGHT CORNER')
            y = y_val
            reduced_mask =  mask_array[0:y_center+l+1, x_center-l:x_len+1]
            reduced_data = data[0:y_center+l+1, x_center-l:x_len+1]
    elif x_center-l < 0:
        # print('HIT LEFT BOUNDARY)
        x = x_val
        reduced_mask = mask_array[y_center-l:y_center+l+1, 0:x_center+l+1]
        reduced_data = data[y_center-l:y_center+l+1, 0:x_center+l+1]
        if y_center+l > y_len:
            # print('HIT TOP LEFT CORNER')
            reduced_mask = mask_array[y_center-l:y_len+1, 0:x_center+l+1]
            reduced_data = data[y_center-l:y_len+1, 0:x_center+l+1]
        if y_center-l < 0:
            # print(f'HIT BOTTOM LEFT CORNER')
            y = y_val
            reduced_mask = mask_array[0:y_center+l+1, 0:x_center+l+1]
            reduced_data = data[0:y_center+l+1, 0:x_center+l+1]
        
    elif y_center+l > y_len:
        # print('HIT TOP BOUNDARY')
        reduced_mask = mask_array[y_center-l:y_len+1, x_center-l:x_center+l+1]
        reduced_data = data[y_center-l:y_len+1, x_center-l:x_center+l+1]
    
    elif y_center-l < 0:
        # print('HIT BOTTOM BOUNDARY')
        y = y_val
        reduced_mask = mask_array[0:y_center+l+1, x_center-l:x_center+l+1]
        reduced_data = data[0:y_center+l+1, x_center-l:x_center+l+1]

    x_mag = (x - x_center) * (x - x_center)
    y_mag = (y - y_center) * (y - y_center)
    r_sq = r * r
    xmax = np.shape(reduced_mask)[1] # dont want to go longer than the reduced mass size
    ymax = np.shape(reduced_mask)[0]
    
    if x_mag + y_mag < r_sq and x_val < xmax and y_val < ymax: 
        # xmax and ymax mean don't go outside mask array size
        # if not doing photometry then mask section
        if photometry == 0:
            reduced_mask[y_val, x_val] = 0
        elif photometry == 1:
            return reduced_data[y_val, x_val]
         
def remove_object(data, mask, y_center, x_center, edge, remove=True):
    start = [y_center, x_center]
    # list of the points that are above the background value
    points_to_visit = [start]
    points_visited = [] # array of the points already visited
    point = start
    no_pixls = 0 # number of pixels in the source
    flux = 0 # addeing up the values of the pixels in the source

    while points_to_visit: # whilst there is something inside the list
        # print(f'number of points left to visit = {len(points_to_visit)}')
        point = points_to_visit.pop(0)
        # print(f'{point=}')

        y, x = point # x and y coordinates of the point in source
        val = data[y][x] # count of particular pixel
        mask[y, x] = 0 # remove the pixel we are looking at

        for direction in (
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ):
            new_point = (y + direction[0], x + direction[1])
            new_y, new_x = new_point
            
            # make sure there are no repeats of points
            if new_point in points_visited:
                # print('found an already visited point')
                continue
            
            points_visited.append(new_point)
            
            # ensure within the range of the data
            if new_y < 0 or new_y >= len(data): 
                # print('found an out of bounds point')
                continue
            if new_x < 0 or new_x >= len(data[0]):
                # print('found and out of bounds point')
                continue
            
            # found a new value to add to the list
            new_val = data[new_y][new_x]
            
            if new_val >= edge: # add the point to sections to look at if useful
                points_to_visit.append(new_point) # add the point to a useful point
                no_pixls += 1 # add 1 to the number of points in source
                flux += val # add the value of the pixel in the source to the flux value
                if remove == True:
                    mask[new_y, new_x] = 0
                    # data *= mask
           
    return no_pixls, flux

def remove_triangle(x_val, y_val, x1, y1, x2, y2, x3, y3, mask_array):
    '''
    Masking triangular shapes due to bleeding  
    x_val and y_val are arrays of x and y coordinates in image
    x1, x2, x3:  x coordinates for corners
    y1, y2, y3 : y coordinates for corners
    mask_array is the updated mask array 
    gradients of each side of the triangle
    '''
    m1 = (x3 - x1) / (y3 - y1)
    m2 = (x3 - x2) / (y3 - y2)
    # constants for each straight line
    c1 = x1 - m1 * y1
    c2 = x2 - m2 * y2
    if y_val > m1*x_val + c1 and y_val < m2*x_val + c2 and x_val > y1:
        mask_array[x_val, y_val] = 0

def remove_rect(x1, x2, y1, y2, mask_array):
    '''
    Masking rectangular shapes due to bleeding  
    x1 and x2 are the x coordinates for rectangular edges 
    y1 and y2 are the y coordinates for rectangular edges 
    mask_array is the updated mask array 
    '''
    mask_array[y1:y2, x1:x2] = 0
    
###### DETECTING SOURCES ######
def find_source(data):
    '''
    Detecting the brightest pixel and therefore centre of source 
    Input : matrix of the image 
    Output: maixmum flux value , location of object centre 
    '''
    max_val = np.max(data)
    locy, locx = np.where(data == max_val)
    # flip x and y because how np.where works
    # matrix i=y, j=x
    return max_val, locy, locx

def source_radius(data, locx, locy, nbins = 500, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650):
    '''
    Determines the source radius 
    Input : 
    Data is the matrix of the image 
    Centre source (locx,locy)
    nbinsis the number of bins used in histogram that dermines local background
    p0 are the initial guesses for the gaussian fit of the count hist 
    [xlim1, xlim2] is the range of x_values in hist 
    Output: radius of object, the value 
    '''
    r = 0 
    locx = int(locx)
    locy = int(locy)
    # pick out an area around the source
    l = 500 # don't make this smaller or you run into issues
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    data_local = data[locx-l:locx+l, locy-l:locy+l]
    background_fit, background_cov = histogram_fit(data_local, nbins, p0=p0, plot=plot, xlim1=xlim1, xlim2=xlim2)
    background = background_fit[1]
    sigma = background_fit[2]
    edge = background + 2 * sigma # anything below this is defined as background
   
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
    return r, background+3*sigma

###### DETECTING ALL SOURCES GIVEN IN AN AREA ######
def detect_circles(input_data, mask):
    y_len = np.shape(input_data)[0]
    x_len = np.shape(input_data)[1]
    # find the background in this section
    background_fit, background_cov = histogram_fit(
        input_data, 
        nbins=4000, 
        title='Background', 
        fit_func=gaussian, 
        plot=False)
    
    A = background_fit[0]
    background = background_fit[1]
    sigma = background_fit[2]
    print(f'{A=}, {background=}, {sigma=}')
    
    stop = 0 # used to stop entire for loop if stuck on a point 
    
    # information we are getting from the searching
    counter = 0
    negative_objects = 0
    xvals = ['object x centers']
    yvals = ['object y centers']
    rs = []
    source_flux = []
    total_flux = ['total flux for aperture']
    back_contrib =['contribution of background to source']
    
    # progress bar
    bar_cls = IncrementalBar #, PixelBar, ShadyBar):
    suffix = '%(percent)d%% [%(elapsed_td)s / %(eta)d / %(eta_td)s]'
    bar = bar_cls(bar_cls.__name__, suffix=suffix, max=380)
    
    # count the number of sources and get the photometry stuff out of it
    for i in range(0, 380): # set a high number of sources
        bar.next()
        # find the source
        max_val, ylocs, xlocs = find_source(input_data)
        # print(f'{max_val=}, {ylocs=}, {xlocs=}')
        
        # stop if got stuck or detecting too faint things
        if stop==1 or max_val < background + 3 * sigma:
            print(f'Counting Stopped Short at {i=}, {counter=}')
            bar.finish()
            break
        
        # print(f'{max_val=}')
        for x, y in zip(xlocs, ylocs):
            # print(x,y)
            
            # reset the values for calculating flux for each object
            total_flux_each = [] # flux of a source + background in area of source
            back_flux_each = [] # total flux of background 
            
            r, local_edge = source_radius(input_data, x, y, nbins=500, p0=[A, background, sigma]) 
            # print(f'{r=}, {local_edge=}')
            range_val = int(10 * r)
            
            if max_val > local_edge:
                if r <= 2: # not countign small objects that could be noise
                    mask[y,x] = 0 # remove 1 random bright pixel
                    input_data *= mask
                    # print(f'found small object {x=}, {y=}')
                    continue
               
                # check that not got stuck on one point
                if x == xvals[-1] and y == yvals[-1]:
                    print('stuck on a point')
                    print(f'{i=}, {x=}, {y=}')
                    stop=1
                    break
                
                # go through each pixel in reduced box
                for px in range(range_val):
                    for py in range(range_val):
                        # determining total flux for fixed aperture
                        # print(f'{px=}, {py=}')
                        current_total_flux = remove_circle(px, py, x, y, r, input_data, mask, y_len=y_len, x_len=x_len, photometry=1)
                        if current_total_flux != None:
                            # print(current_total_flux)
                            total_flux_each.append(current_total_flux)
                        # remove the source now obtained information
                        remove_circle(px, py, x, y, r, input_data, mask, y_len=y_len, x_len=x_len)
                # print(f'normal object_masked')
                input_data *= mask
                
                # see_image(input_data)
                
                # go through each pixel now the object has been removed
                for px2 in range(range_val):
                    for py2 in range(range_val):
                        # print(f'{px2=}, {py2=}')
                        current_background_flux = remove_circle(px2, py2, x, y, r+10, input_data, mask, y_len=y_len, x_len=x_len, photometry=1)
                        if current_background_flux != None:
                            # print(current_background_flux)
                            back_flux_each.append(current_background_flux)
                        
            elif max_val <= local_edge:
                # print(f'object too faint, {i=}, {max_val=}')
                mask[y, x] = 0
                input_data *= mask
                continue
            
            # make sure done required masking
            input_data *= mask
            
            annular_no_pixls = len(back_flux_each)
            source_no_pixls = len(total_flux_each)
            
            back_no_pixls = annular_no_pixls - source_no_pixls # number of pixels in the ring around source
            
            # background flux per pixel (background flux density)
            back_density = int(sum(back_flux_each)) / back_no_pixls # denisty of background
            # find the contribution of background flux to source flux
            back_contrib_each = back_density * source_no_pixls
            # back_contrib.append(back_contrib_each)
            
            source_flux_each = sum(total_flux_each) - back_contrib_each
            if source_flux_each < 0:
                negative_objects += 1
                # print('NEGATIVE SOURCE FLUX')
                # print(f'{i=}, {x=}, {y=}, {r=}')
                # print(f'{annular_no_pixls=}, {back_no_pixls=}, {source_no_pixls=}')
                # print(f'{sum(back_flux_each)=}, {back_density=}, {sum(total_flux_each)=}, {back_contrib_each=}, {source_flux_each=} ')
                continue
            
            else: # we don't append negative fluxes since this means that we have found something other than a source
                source_flux.append(source_flux_each)
                xvals.append(x)
                yvals.append(y)
                rs.append(r)
                counter += 1 # counting the number of detected objects  
                
    return counter, source_flux

def detect_sources(input_data, mask):
    stop = 0 # used to stop entire for loop if stuck on a point 
    counter = 0
    y_vals = ['source y centers']
    x_vals = [' source x centers']
    source_fluxes = []
    annular_fluxes = []
    
    y_len = np.shape(input_data)[0]
    x_len = np.shape(input_data)[1]
    
    # find the background in this section
    background_fit, background_cov = histogram_fit(
        input_data, 
        nbins=4000, 
        title='Background', 
        fit_func=gaussian, 
        plot=False)
    
    A = background_fit[0]
    background = background_fit[1]
    sigma = background_fit[2]
    edge = background + 3.5 * sigma
    annular_edge = background + 2 * sigma
    print(f'{A=}, {background=}, {sigma=}')
    
    max_fev = 400
    # progress bar
    bar_cls = IncrementalBar #, PixelBar, ShadyBar):
    suffix = '%(percent)d%% [%(elapsed_td)s / %(eta)d / %(eta_td)s]'
    bar = bar_cls(bar_cls.__name__, suffix=suffix, max=max_fev)
    
    for i in range(max_fev):
        # print(f'{i=}')
        bar.next()
        max_val, locy, locx = find_source(input_data)
        # print(f'{max_val=}, {locy=}, {locx=}')
        
        if max_val <= edge:
            print(f'counting stopped short at {i=}')
            bar.finish()
            break
        
        elif max_val >= edge:     
            for y_center, x_center in zip(locy, locx):
                y_center, x_center = int(y_center), int(x_center)
                r, local_edge = source_radius(input_data, x_center, y_center, p0=[A,background,sigma])
                if y_center == y_vals[-1] and x_center == x_vals[-1]:
                    print(f' stuck on a point {i=}. {y_center=}. {x_center=}')
                # annular_pixls, annular_flux = remove_object(input_data, mask, y_center, x_center, annular_edge, remove=False)
                for px in range(3*r):
                    for py in range(3*r):
                        annular_flux_current = remove_circle(px, py, x_center, y_center, r+5, input_data, mask, y_len, x_len, 1)
                        if annular_flux_current != None:
                            # print(annular_flux_current)
                            annular_fluxes.append(annular_flux_current)
                
                source_pixls, total_flux = remove_object(input_data, mask, y_center, x_center, edge, remove=True)
                
                if source_pixls >= 15: # only add big enough sources
                    counter += 1
                    y_vals.append(y_center)
                    x_vals.append(x_center)
                    
                    # finding flux of source without background contribution
                    annular_flux = np.sum(annular_fluxes)
                    annular_pixls = len(annular_fluxes)
                    # print(f'{annular_flux=}, {annular_pixls=}')
                    back_flux = annular_flux - total_flux
                    back_pixls = annular_pixls - source_pixls
                    back_density = back_flux / back_pixls
                    back_contrib = back_density * source_pixls # contribution of background to the source 
                    source_flux = total_flux - back_contrib
                    source_fluxes.append(source_flux) # add this source the list of all sources
                    # print(f'{back_pixls=}, {source_flux=}')
                input_data *= mask

    return counter, source_fluxes
    
    









    