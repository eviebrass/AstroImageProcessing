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
import CatalogueSaving as s
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

data_count = data_no_bleed#[1400:1600, 1400:1600] # data using to count stars
mask_count = mask_array#[1400:1600, 1400:1600] # array we will update to count stars

# information we are getting from the searching
counter = 0
stop = 0 # used to stop entire for loop if stuck on a point 
xvals = ['object x centers']
yvals = ['object y centers']
rs = []

total_flux =['total flux for aperture']
local_back =['Finding local background']
for i in range(0, 10000000): # testing by fixing the number of sources we want to count
    # find the source
    max_val, ylocs, xlocs = func.find_source(data_count)
    
    # stop if got stuck or detecting too faint things
    if stop==1 or max_val < 3400:
        print('Counting Stopped Short')
        break
    
    # print(f'{max_val}, {xlocs=}, {ylocs=}')
    for x, y in zip(xlocs, ylocs):
        # print(x,y)
        r, local_edge = func.source_radius(data_count, x, y,nbins=500) 
        # print(f'{r=}, {local_edge=}')
        if max_val > local_edge:
            if r <= 3: # not countign small objects that could be noise
                mask_count[y,x] = 0 # remove 1 random bright pixel
                data_count *= mask_count
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
            
            total_flux_each = [] # total flux for given aperture
            back_flux_each = [] # total flux for background 
            
            for px in range(150):
                for py in range(150):
                    # determining total flux for fixed aperture
                    if func.remove_circle(px, py, x, y, r, mask_count, photometry=1) == True:
                        total_flux_each.append(data_count[px,py])
                    # determing the number of pixel for object 
                    pixl_source = func.remove_circle(px, py, x, y, r+10, mask_count, pixl=1)
                    # masking the object 
                    func.remove_circle(px, py, x, y, r, mask_count) # make sure flip x and y here
            
            data_count *= mask_count
            
            for px in range(200):
                for py in range(200):
                    if func.remove_circle(px, py, x, y, r+10, mask_count, photometry=1) == True:
                        back_flux_each.append(data_count[px,py]) #adding flux for each background to list
        
        elif max_val <= local_edge:
            print(f'object too faint, {i=}, {max_val=}')
            mask_count[y-1:y+1,x-1:x+1] = 0
            data_count *= mask_count
            continue
        
        data_count *= mask_count
        #finding total flux for set aperture
        total_flux.append(np.sum(total_flux_each))
        
        #determing background count per pixel 
        back_count_pixl = np.sum(back_flux_each)/len(back_flux_each)
        #determing the local background by multiplying flux by pixels in aperture 
        local_back_each = back_count_pixl * pixl_source 
        local_back.append(local_back_each)
#%%
fits.writeto('removing_objects.fits', data_count, overwrite=True)

print('Source Counting is Finished')

# determing the source flux 
source_flux = np.array(total_flux[1:]) - np.array(local_back[1:])

end = time.perf_counter()

total_time = (end - start)/60 # find the amount of time it takes to run.
print(f'{total_time = } minutes')

#%%
# =============================================================================
# 5.5 Calibrating the fluxes. 
# Converting instrumental counts to source magnitude
# =============================================================================

# converting counts into instrumental magnitude 
inst_mag = -2.5 * np.log10(source_flux)

# converting instrumental arguments into calibrated magnitudes 
mag = point_0 + inst_mag # point_0 defined at the top 
# uncertainty in magnitude 
plt.hist(mag,bins=200)
plt.show()

# =============================================================================
# 6.1 Comparison to liner fit
# =============================================================================

# determining the numbner of sources detected brighter than a magntitude limit 
limit = 3631e-26 # AB system magntiutde
pixl_deg = np.pi / (0.258 * 648000) # conversion factor from pixl to deg 

area = np.shape(data)[0] * np.shape(data)[1] * pixl_deg * pixl_deg # finding area of whole image
# range of magnitudes considered 

mag_range = np.arange(9,np.max(mag),step=0.1)

# determining number of objects with brightness less than given magnitude
n_m = [] # number of objects
for mag_max in  mag_range: # maximum magnitude
    count_obj_greater = 0 # number of objects with magnitude greater than each mag
    for each_mag  in mag: # magnitude of each object 
        if each_mag > mag_max: 
            count_obj_greater += 1 
    n_m.append(count_obj_greater) 
        
N_m = np.array(n_m)/area # density of number of objects

# taking the log of the magnitude 
log_N_m = np.log10(N_m)

# fitting to find gradient 
pars,cov = np.polyfit(mag_range,log_N_m,1,cov=1)
print('Gradient %.3e +/- %.3e' % (pars[0],cov[0,0]))

plt.plot(mag_range,log_N_m,'x')
plt.plot(mag_range,pars[0]*mag_range + pars[1],color='black')
plt.xlabel('m')
plt.ylabel('log(N(m))')
plt.grid()
plt.show()












