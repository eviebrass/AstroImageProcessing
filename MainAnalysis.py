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

#%% Don't run this file all the time
# Check the contents of the FITS file
hdulist = fits.open('A1_mosaic.fits')
with open('hdulist.pickle', 'wb') as f:
    pickle.dump(hdulist[0].data, f)
#%%
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

#%% Only run this cell if changing the masked parts
###### IMAGE MASKING ######
mask_array = np.ones(np.shape(data)) # initially all 1's

for px in range(4172):
    for py in range(2135):
        # remove circle parts
        func.remove_circle(px, py, 1200, 3000, 200, mask_array)
        func.remove_circle(px, py, 558, 3105, 40, mask_array)
        func.remove_circle(px, py, 757, 2555, 35, mask_array)
        func.remove_circle(px, py, 687, 2066, 40, mask_array)
        func.remove_circle(px, py, 1917, 3540, 25, mask_array)
        # remove triangular parts
        func.remove_triangle(px, py, 1161, 0, 1275, 0, 1222, 57, mask_array)
        func.remove_triangle(px, py, 1121, 103, 1287, 103, 1227, 171, mask_array)
        func.remove_triangle(px, py, 1124, 216, 1303, 216, 1220, 274, mask_array)

# removing any rectangular points
# rectangles associated with stars
mask_array[0:-1, 1210:1230] = 0 
mask_array[2980:3198, 555:561] = 0
mask_array[2484:2616, 753:760] = 0
mask_array[2003:2136, 684:691] = 0
mask_array[3487:3582, 1915:1925] = 0
# # rectangles below triangles
mask_array[95:104, 803:1487] = 0 
mask_array[207:217, 887:1434] = 0 
# other random points
mask_array[207:210, 810:826] = 0
mask_array[205:232, 819:822] = 0
mask_array[114:135, 1424:1431] = 0
    
data_no_bleed = mask_array * data
fits.writeto('no_bleed_data.fits', data_no_bleed, overwrite=True)

with open('data_no_bleed.pickle', 'wb') as f:
    pickle.dump(data_no_bleed, f)

print('removing bleeds is complete')

#%%### DETECTING SOURCES ######

with open('data_no_bleed.pickle', 'rb') as f:
    data_no_bleed = pickle.load(f)

background_edge = background_count + 3 * background_spread # Three sigma above the wanted value

def find_source(data_current):
    max_val = np.max(data_current)
    print(f'{max_val=}')
    locx, locy = np.where(data_current == max_val)
    locx = int(locx)
    locy = int(locy)
    r = 12
    # take an area for the background calculations
    l = 50 # length to go either side of the source
    data_flat = np.ravel(data_current[locx-l:locx+l, locy-l:locy+l]) # picking out a square around the source
    hist_y, hist_edges = np.histogram(data_flat, bins=2000)  
    hist_centers = 0.5*(hist_edges[1:] + hist_edges[:-1])
    hist_error = np.sqrt(hist_y)
    hist_fit, hist_cov = curve_fit(gaussian, hist_centers, hist_y, p0=[7e6,3420,18])
    x_hist_fit = np.linspace(3300, 3650, 1000)
    plt.plot(x_hist_fit, gaussian(x_hist_fit, *hist_fit), color='black', label = 'gaussian fit')
    plt.errorbar(hist_centers, hist_y, yerr= hist_error,color='red', fmt='x')
    plt.hist(data_flat, bins=2000, label ='Pixel Counts')
    plt.xlim(3300,3650)
    plt.legend()
    plt.show()
    background = hist_fit[0]
    sigma = hist_fit[1]
    edge = background + 3 * sigma # anything below this is defined as background
    # find the radius of the detected star
    data_scan = data_current[locx:locx+100, locy] # limit the region that we are searching
    for x in range(len(data_scan)): # pick out the points along y to find radius
        if data_scan[x] < edge:
            r = x
            break
    return locx, locy, r, max_val

data_count = data_no_bleed # data using to count stars
mask_count = mask_array # array we will update to count stars

# information we are getting from the searching
counter = 0
xlocs = ['object x centers']
ylocs = ['object y centers']
for i in range(0, 2): # testing by fixing the number of sources we want to count
    x_current, y_current, r_current, max_val_current = find_source(data_count)
    if max_val_current > background_edge and r_current > 3: # not counting really small things
        xlocs.append(x_current)
        ylocs.append(y_current)
        counter += 1 # counting the number of detected objects
        # print(i)
        # print(f'{x_current=}, {y_current=}, {r_current=}')
        for px in range(4172):
            for py in range(2135):
                if remove_circle(px, py, y_current, x_current, r_current) == True: # make sure that you flip the x and y in this case
                    mask_count[px, py] = 0
        data_count = mask_count * data_count

data_count = mask_count * data_count

fits.writeto('removing_objects.fits', data_count, overwrite=True)

# pd.dataFrame()

# trial saving a catalogue

print('Source Counting is Finished')

#%%### GET RID OF DATA THAT WE DEEM TO BE BACKGROUND ######

background_cut_off = background_count + 3 * background_spread # Three sigma above the wanted value
# remove points below our maximum value

# remove points below our maximum value
min_i = np.array(np.where(data_no_bleed <= background_cut_off))
x_min_i = min_i[0,:]
y_min_i = min_i[1,:]
for j in range(len(x_min_i)):
        mask_array[x_min_i[j],y_min_i[j]] = 0
        
data_final = mask_array * data_no_bleed
fits.writeto('data_no_background.fits', data_final, overwrite=True)
plt.imshow(data_final, cmap='gray')
plt.colorbar()
plt.show()

#%% PRINTING THE USEFUL VALUES
print(f'{background_count= :.5f}')
print(f'{background_spread= :.5f}')











