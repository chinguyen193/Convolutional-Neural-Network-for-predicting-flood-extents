"""
Created on Tue Feb  7 22:29:24 2023

@author: ngu204
"""
# this code is to calculate the Potential Connection Length  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import rc
from matplotlib import cm
import os
import time
import rioxarray as rxr
from skimage import measure
from scipy.ndimage.morphology import binary_erosion
from rasterio.merge import merge


#%%
# input: dem, maximun flooded area and a raster to assign the input location as 1
# output: potential flow length

def potential_connection_length(dem, max_flood, cc):


    save = np.zeros((dem.shape[0],dem.shape[1])) 
    
    # calculate slopes in 4 direction
    slope1 = dem - np.roll(dem,-1)
    slope2 = dem - np.roll(dem,-1, axis = 0)
    slope3 = dem - np.roll(dem,1)
    slope4 = dem - np.roll(dem,1, axis = 0)
    
    n = 0
    
    cc_save = np.zeros((dem.shape[0],dem.shape[1]))
    
    start_time = time.time()
    
    while np.array_equal(cc, cc_save) == False :
        
        #start_time = time.time()
        #main()
        
        n = n+1    
        # save the last layer before update
        cc_save = np.array(cc)
        #
        cc_bound = cc - binary_erosion(cc)    
        save = np.zeros((dem.shape[0],dem.shape[1]))
        save[max_flood==1]=0.5
        save[cc_bound>0] = save[cc_bound>0]+0.5
        
        #consider only cells within flood map and cc > 0 
        index = np.where(save==1)
        x = np.array(index[0])
        y = np.array(index[1])
        
        for count in range(len(x)):
            j = x[count]
            i = y[count]
            # check order of the slope 
            slope = np.array([slope1[j,i],slope2[j,i],slope3[j,i],slope4[j,i]])
            sorter = np.argsort(np.sort(slope))
            order = sorter[np.searchsorted(np.sort(slope), slope, sorter=sorter)]
            order = order + 1
            
            # assign value if the value is smaller than the current value
            if i < dem.shape[1]-1:
                cc[j,i+1] = cc[j,i] + order[0] if cc[j,i] + order[0] <= cc[j,i+1] or cc[j,i+1] == 0 else cc[j,i+1] # 1            
            if j > 0:
                cc[j-1,i] = cc[j,i] + order[1] if cc[j,i] + order[1] <= cc[j-1,i] or cc[j-1,i] == 0 else cc[j-1,i] # 2
            if i > 0:
                cc[j,i-1] = cc[j,i] + order[2] if cc[j,i] + order[2] <= cc[j,i-1] or cc[j,i-1] == 0 else cc[j,i-1] # 3
            if j < dem.shape[0]-1:
                cc[j+1,i] = cc[j,i] + order[3] if cc[j,i] + order[3] <= cc[j+1,i] or cc[j+1,i] == 0 else cc[j+1,i] # 4
            
            
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return cc


#%% Plot the probability density function 

#input:
# CNN_fm: binary flood extent
# PCL_map: the PCL map calculated using the 'potential_connection_length' function

def Plot_relative_frequency(CNN_fm, PCL_map, num_bins = 50):

    # assign the flow length associate with with the flood map, ignore NoData values
    cc = np.where(CNN_fm>=0.5,PCL_map,-9999) 
    
    # calculate the probability distribution:
    
    fig0, ax0 = plt.subplots(figsize=(6, 5))
    n, bins, patches = ax0.hist(cc[cc>0], num_bins, density=True)
    bin_width = bins[1]-bins[0]
    bin_values = (bins[1:]+bins[0:-1])/2
    
    # plot the pdf 
    fig, ax = plt.subplots()
    ax.bar(bin_values*25/1000,n*bin_width,width = 0.95)
    ax.set_xlabel('Potential flow length (km)')
    ax.set_ylabel('Probability')
    ax.tick_params(which='major', length=5)
    
    
    # set the title
    mu = np.mean(cc[cc>0])  # mean of distribution
    sigma = np.std(cc[cc>0])  # standard deviation of distribution
    max_len = np.amax(cc[cc>0])
    
    title  = '$\mu$ = ' + str(np.round(mu,1)) + ',' + '$\sigma$ = ' + str(np.round(sigma,1))
    
    #title  = '$\max$ = ' + str(np.round(max_len,1)) + ',' + '$\mu$ = ' + str(np.round(mu,1)) + ',' + '$\sigma$ = ' + str(np.round(sigma,1))
    ax.set_title(title)
    
    plt.tight_layout()
    
    return fig
#%% 
# read the DEM and the maximun flood extent
dem = rxr.open_rasterio('DEM_25m.tif')
max_flood = rxr.open_rasterio('Flooded_area_edit2.asc')

# convert into numpy arrays
dem = np.array(dem[0])
max_flood = np.array(max_flood[0])

# create a layer to assign input points
cc = np.zeros((dem.shape[0],dem.shape[1]))
cc[55:65,0] = 1

# calculate PCL map
PCL_map = potential_connection_length(dem, max_flood, cc)































