#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:20:04 2022
@author: eviebrass
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
from scipy.optimize import curve_fit
from astropy.io import ascii # for saving catalogue as acsii fie
from astropy.table import Table
import FunctionsFile as func
import csv

point_0 = 2.53e1
point_0_un = 2e-2

# # Check the contents of the FITS file
# hdulist = fits.open('A1_mosaic.fits')
# with open('hdulist.pickle', 'wb') as f:
#     pickle.dump(hdulist[0].data, f)


with open('hdulist.pickle', 'rb') as f:
    data = pickle.load(f)

#### SEE THE IMAGE ######
func.see_image(data)

#### CROPPING THE EDGES OF THE IMAGE ######
edge_val = data[0][0]
# find the height of the border
for i in range(len(data[100][:])):
    if data[100][i] == edge_val:
        continue
    elif data[100][i] != edge_val:
        height = i + 100
        break
# find the width of the border
for i in range(len(data[:][100])):
    if data[i][100] == edge_val:
        continue
    elif data[i][100] != edge_val:
        width = i + 100
        break

#cropping picture 
#data = data[width:4610-width][height:2569-height]
data = data[width:4610-width,height:2569-height]

# save the new file data
fits.writeto('cropped_image.fits', data, overwrite=True)

#### SEE THE IMAGE ######
func.see_image(data)

global_background_fit, global_background_cov = func.histogram_fit(
    data, 
    nbins=4000, 
    title='Global Background', 
    fit_func=func.gaussian, 
    plot=True)

global_background = global_background_fit[1]

#%%
###### IMAGE MASKING ######
import time

mask_array = np.ones(np.shape(data)) # initially all 1

# remove circular parts
for px in range(600):
    for py in range(600):
        # remove circle parts
        func.remove_circle(px, py, 1220, 3000, 240, mask_array)
        func.remove_circle(px, py, 558, 3105, 45, mask_array)
        func.remove_circle(px, py, 757, 2555, 40, mask_array)
        func.remove_circle(px, py, 687, 2066, 45, mask_array)
        func.remove_circle(px, py, 1917, 3540, 30, mask_array)
# remove triangular parts
for px in range(300):
    for py in range(1100, 1350):
        func.remove_triangle(px, py, 1161, -1, 1275, -1, 1222, 57, mask_array)
        func.remove_triangle(px, py, 1121, 103, 1287, 103, 1227, 171, mask_array)
        func.remove_triangle(px, py, 1124, 216, 1303, 216, 1220, 274, mask_array)

# removing any rectangular points
# rectangles associated with stars
mask_array[:, 1205:1240] = 0 
mask_array[2980:3198, 555:564] = 0
mask_array[2484:2616, 753:760] = 0
mask_array[2003:2136, 684:691] = 0
mask_array[3487:3582, 1913:1925] = 0
# # rectangles below triangles
mask_array[95:104, 803:1487] = 0 
mask_array[207:217, 887:1434] = 0 
# other random points
mask_array[207:210, 810:826] = 0
mask_array[205:232, 819:822] = 0
mask_array[114:135, 1424:1431] = 0
# removing noisy section in top left corner
mask_array[4090:, :90] = 0

data_no_bleed = mask_array * data
fits.writeto('no_bleed_data.fits', data_no_bleed, overwrite=True)

print('removing bleeds is complete')

###### DETECTING SOURCES ######

start = time.perf_counter() # start time of the code

data_count = data_no_bleed[500:1000, 500: 1000] #[1400:1600, 1400:1600] # data using to count stars
mask_count = mask_array[500: 1000, 500:1000] #[1400:1600, 1400:1600] # array we will update to count stars

#obtaining soruce data for catalogue 

counter, xvals, yvals, rs, total_flux, annular_back = func.detect_sources(data_count, mask_count)


fits.writeto('removing_objects.fits', data_count, overwrite=True)

print('Source Counting is Finished')

end = time.perf_counter()

total_time = (end - start)/60 # find the amount of time it takes to run.
print(f'{total_time = } minutes')

#%% Writing an ASCII Table using ascii.write() 

#defining data
data = {'Position x': xvals[1:],
        'Position y': yvals[1:],
         'Radius': rs[1:],
         'Counts Source': total_flux[1:] ,
         #'Un counts': un_source_flux,
         'Source Background': annular_back[1:]}
         #'Source back un': annular_back_un,}

# writing a file with data 
ascii.write(data,'catalogue_ap=100.csv',format='csv',overwrite=1)

#%%
# =============================================================================
# 5.5 Calibrating the fluxes. 
# Converting instrumental counts to source magnitude
# =============================================================================
#reading the data 
x_val,y_val, r, total_flux,source_back = np.loadtxt('catalogue_ap=5.csv', delimiter= ',', skiprows = 2 , unpack=1)

# determing the source flux 
source_flux = total_flux - source_back
print(source_flux)

# converting counts into instrumental magnitude 
inst_mag = -2.5 * np.log10(source_flux)

# converting instrumental arguments into calibrated magnitudes 
mag = point_0 + inst_mag # point_0 defined at the top 
# for point in mag:
#     if mag >= 13:
          
# uncertainty in magnitude 
mag_fit, mag_cov, mag_centers, mag_freq = func.histogram_fit(mag, nbins=8, fit_func=func.exponential, p0=[1,1.1,8.4], plot=True, xlim1=0, xlim2=15, log_plot=True)
# plt.hist(mag,bins=15)
plt.show()

# =============================================================================
# 6.1 Comparison to liner fit
# =============================================================================

# determining the numbner of sources detected brighter than a magntitude limit 
limit = 3631e-26 # AB system magntiutde
pixl_deg = np.pi / (0.258 * 648000) # conversion factor from pixl to deg 

area = np.shape(data)[0] * np.shape(data)[1] * pixl_deg * pixl_deg # finding area of whole image

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
    y_label = 'log$_{10}$(N(m))', 
    data_colour = 'red', 
    fit_colour = 'black', 
    fit_func = func.linear)
plt.plot(mag_centers, log_N, 'x')

print(f'gradient ={ log_fit[0]:.3f}')
