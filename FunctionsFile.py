import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

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
def histogram_fit(data, nbins, title='', fit_func=gaussian, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650, log_plot=False):
# =============================================================================
# Plotting histogram to determine gaussian fit of flux hist 
# nbins is the number of bins  
# fit_func is the gaussian fucntion 
# plot shows image is true
# x_lims1,2 are x-axis range
# log_plot determines retuns hist with uncertainty if true
# ===========================================================
    data_flat = np.ravel(data)
    hist_y, hist_edges = np.histogram(data, bins=nbins)
    hist_centers = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_error = np.sqrt(hist_y)
    hist_fit, hist_cov = curve_fit(fit_func, hist_centers, hist_y, p0)
    if plot == True:
        x_hist_fit = np.linspace(xlim1, xlim2, 1000)
        plt.plot(x_hist_fit, fit_func(x_hist_fit, *hist_fit), color='black', label = 'gaussian fit')
        plt.errorbar(hist_centers, hist_y, yerr= hist_error, color='red', fmt='x')
        plt.hist(data_flat, bins=nbins, label ='Pixel Counts')
        plt.xlim(xlim1,xlim2)
        plt.title(title)
        plt.legend()
        plt.show()
    if log_plot == False:
        return hist_fit, hist_cov
    elif log_plot == True:
        return hist_fit, hist_cov, hist_centers, hist_y

###### IMAGE MASKING ######
def remove_circle(x_val, y_val, x_center, y_center, r, mask_array, photometry=0, pixl=0):
# =============================================================================
# Masking objects assuming circular shapes 
# x_val and y_val are all the coordiante values in image 
# (x_centre,y_centre) are centre coordinates for object 
# r is the radius of the objects  
# mask_array is the most up to date mask array 
# photometry obtains the flux of each pixel in aperture if true
# pixl counts the number of pixels in given aperture if true 
# ===========================================================
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
# =============================================================================
# Masking triangular shapes due to bleeding  
# x_val and y_val are arrays of x and y coordinates in image
# x1, x2, x3:  x coordinates for corners
# y1, y2, y3 : y coordinates for corners
# mask_array is the updated mask array 
# ===========================================================

    # gradients of each side of the triangle
    m1 = (x3 - x1) / (y3 - y1)
    m2 = (x3 - x2) / (y3 - y2)
    # constants for each straight line
    c1 = x1 - m1 * y1
    c2 = x2 - m2 * y2
    if y_val > m1*x_val + c1 and y_val < m2*x_val + c2 and x_val > y1:
        mask_array[x_val, y_val] = 0

def remove_rect(x1, x2, y1, y2, mask_array):
# =============================================================================
# Masking rectangular shapes due to bleeding  
# x1 and x2 are the x coordinates for rectangular edges 
# y1 and y2 are the y coordinates for rectangular edges 
# mask_array is the updated mask array 
# ===========================================================
    mask_array[y1:y2, x1:x2] = 0
    
###### DETECTING SOURCES ######
def find_source(data):
# =============================================================================
# Detecting the brightest pixel and therefore centre of source 
# Input : matrix of the image 
# Output: maixmum flux value , location of object centre 
# =============================================================================
    max_val = np.max(data)
    locy, locx = np.where(data == max_val)
    # flip x and y because how np.where works
    # matrix i=y, j=x
    r = 0
    l = 50 # length to go either side of the source
    return max_val, locy, locx

def source_radius(data, locx, locy, nbins = 1000, p0=[7e6,3420,18], plot=False, xlim1=3300, xlim2=3650):
# =============================================================================
# Determines the source radius 
### Input : 
# Data is the matrix of the image 
# Centre source (locx,locy)
# nbinsis the number of bins used in histogram that dermines local background
# p0 are the initial guesses for the gaussian fit of the count hist 
# [xlim1, xlim2] is the range of x_values in hist 
### Output: radius of object, the value 
# =============================================================================
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

###### DETECTING ALL SOURCES GIVEN IN AN AREA ######
def detect_sources(data, mask):
# =============================================================================
# Detects the objects and masks them. Also returns flux of background and source 
# Data is the matrix of the image 
# mask is the mask array 
# Output: number of objects, location, radius, total flux of source, 
# annular/background flux per pixel, total flux of source - background 
# =============================================================================
    
    # find the background in this section
    background_fit, background_cov = histogram_fit(
        data, 
        nbins=4000, 
        title='Background', 
        fit_func=gaussian, 
        plot=False)

    background = background_fit[1]
    # print(f'{background=}')
    
    stop = 0 # used to stop entire for loop if stuck on a point 
    
    # information we are getting from the searching
    counter = 0
    # object x centers
    xvals = [0]
    # objects y centres
    yvals = [0]
    # objects radius
    rs = [0]
    
    # total flux of object 
    total_flux =[0]
    # background flux of object 
    annular_back =[0]
    
    # count the number of sources and get the photometery stuff out of it
    for i in range(0, 50): # set a high number of sources
        # find the source
        max_val, ylocs, xlocs = find_source(data)
        
        # stop if got stuck or detecting objects that are too faint 
        if stop==1 or max_val < background:
            print('Counting Stopped Short')
            break
        
        # print(f'{max_val}, {xlocs=}, {ylocs=}')
        
        for x, y in zip(xlocs, ylocs):
            # print(x,y)
            r, local_edge = source_radius(data, x, y,nbins=500) 
            # print(f'{r=}, {local_edge=}')
            if max_val > local_edge: #ensures background is not considered as source
                if r <= 3: # not counting small objects that could be noise
                    mask[y,x] = 0 # remove 1 random bright pixel
                    data *= mask
                    # print(f'found small object {x=}, {y=}')
                    continue
               
                # check that not got stuck on one point
                if x == xvals[-1] and y == yvals[-1]:
                    print('stuck on a point')
                    print(f'{i=}, {x=}, {y=}')
                    stop=1
                    break
                
                xvals.append(x)
                yvals.append(y)
                rs.append(r)
                counter += 1 # counting the number of detected objects
                print(f'{counter=}, {max_val=}')
                
                source_flux_each = [] # flux of each pixel in  source 
                back_flux_each = [] # flux of each pixel in annular region 
                
                for px in range(150):
                    for py in range(150):
                        # determining total flux of the object 
                        if remove_circle(px, py, x, y, r, mask, photometry=1) == True:
                            source_flux_each.append(data[px,py])
                        # determing the number of pixel in object 
                        pixl_source = remove_circle(px, py, x, y, r, mask, pixl=1)
                        # masking the object 
                        remove_circle(px, py, x, y, r, mask) 
                
                data *= mask
                
                for px in range(200):
                    for py in range(200):
                        # determining the flux of annualar region around object 
                        if remove_circle(px, py, x, y, r+10, mask, photometry=1 == True):
                            back_flux_each.append(data[px,py]) #adding flux for each pixel in background to list
            
            elif max_val <= local_edge:
                print(f'object too faint, {i=}, {max_val=}')
                mask[y, x] = 0
                data *= mask
                continue
            
            #data *= mask
            
        #assummng flux of each pixel in object to find the flux for each object 
        total_flux.append(np.sum(source_flux_each)) 
            
            
         #determing background flux per pixel in background 
         back_count_pixl = np.sum(back_flux_each)/len(back_flux_each) 
         #determing the background in source by multiplying background flux per 
         # pixels by the number of pixels in the objects 
         annular_back_each = back_count_pixl * pixl_source 
         #adding the total background contribution for each source to list for all objects 
         annular_back.append(annular_back_each)
        
    return counter, xvals, yvals, rs, total_flux, back_count_pixl, annular_back
    
