# Code for testing 
import numpy as np
import matplotlib.pyplot as plt
import random

# defining simple data sets for testing
empty_test = np.zeros((4172,2135)) # empty
same_val_test = np.ones((4172,2135)) # same value
noise_test = np.random.rand(4172,2135) # noise

# defining a function that outputs gaussain blob to simular Gaussian blobs
def gauss_2d(size,x_offset,y_offset,sigma,ampl):
    # for recommended shape let sigma = 1 
    x = np.arange(0,size)
    y = np.arange(0,size)
    xx,yy= np.meshgrid(x,y)
    xx= xx-x_offset
    yy= yy -y_offset
    sigma =  (sigma/0.258)*2.35 #used 1 arsec FWHM, convert to pixel then sd
    power = (xx*xx) +(yy*yy)
    denom = 1/(2*np.pi*sigma*sigma)
    return ampl*denom* np.exp(-power/(2*sigma*sigma))  

# defining an unrotated elipse 
def elipse(size,x_offset,y_offset,sigma_x,sigma_y,theta,ampl):
    x = np.arange(0,size)
    y = np.arange(0,size)
    xx,yy= np.meshgrid(x,y)
    xx_new= xx-x_offset
    yy_new= yy -y_offset
    cos = np.cos(theta)
    sin = np.sin(theta)
    a = (cos*cos)/(2*(sigma_x**2)) + (sin*sin)/(2*(sigma_y**2))
    b = -np.sin(2*theta)/(4*(sigma_x**2)) + np.sin(2*theta)/(4*(sigma_y**2))
    c = (sin*sin)/(2*(sigma_x**2)) + (cos*cos)/(2*(sigma_y**2))
    power = a*xx_new*xx_new + 2*b*xx_new*yy_new + c*yy_new*yy_new
    return ampl *np.exp(-power)

n_objects = 10 # number of objects  -1
size_m = 1000 # width/height of image as square

# generating random location array
random.seed(20)
loc_ran = [[random.randint(0,size_m) for i in range(n_objects)] for j in range(2)]
test_obj = [] #2d gaussian
test_elip = [] #2d ellips

# randomising the radii of objects from 0  to 10 arcseconds
sigma_rand = [random.uniform(1,4) for i in range(n_objects)]

# creating gaussian object 
for i in range(n_objects): # same as i value
    obj_n = gauss_2d(size_m,loc_ran[0][i],loc_ran[1][i],sigma_rand[i],1000)
    test_obj.append(obj_n)

# standard sigma size 
sigma =  (1/0.258)*2.35 # used 1 arsec FWHM, convert to pixel then sd

# generating random angle array for elipse 
angle_rand = [random.uniform(0,np.pi) for i in range(n_objects)]

# creating roated elipses 
for i in range(n_objects): # same as i value
    elip_n = elipse(size_m,loc_ran[0][i],loc_ran[1][i],sigma,sigma/2,angle_rand[i],1000)
    test_elip.append(elip_n)
    
obj_final= sum(test_obj) # circular overlap image 
elip_final = sum(test_elip) # eliptic overlap image 

# generating noise for the background for gaussian 
noise_obj = np.random.normal(0,np.max(obj_final)*0.05,size=(size_m,size_m))
obj_noise = obj_final + noise_obj  # adding noise to gaussian objects

# generating noise for the background for elipse 
noise_elip = np.random.normal(0,np.max(elip_final)*0.1,size=(size_m,size_m))
elip_noise = elip_final + noise_elip 

# plotting new image 
plt.title('Gaussian with noise')
plt.imshow(obj_noise,cmap='gray')
plt.show()

plt.title('Elipse with noise ')
plt.imshow(elip_noise,cmap='gray')
plt.show()
