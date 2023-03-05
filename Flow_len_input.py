# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:29:24 2023

@author: ngu204
"""
# this code is to calculate potential flow length counting from the input cell  

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

#%% save data

def WriteRaster(filename, raster, xll = 1466225.000000000000, yll = -3328900.000000000000, cell_size = 25, NoData_value = -9999):  
    file = open(filename, 'w+')
    nrows, ncols = np.shape(raster)
    np.savetxt(filename, raster, fmt='%.1f', delimiter = ' ')
    #save the array 
    data_old = file.readlines()
    #add the head lines to the file
    file = open(filename, 'w+')
    file.writelines('ncols \t %.0f \n' % ncols)
    file.writelines('nrows \t %.0f \n' % nrows )
    file.writelines('xllcorner \t %f \n' % xll)
    file.writelines('yllcorner \t %f \n' % yll)
    file.writelines('cellsize \t %f \n' % cell_size)
    file.writelines('NODATA_value \t %f \n' % NoData_value)
    file.writelines(data_old)
    #ascii.write(raster, filename, overwrite=True)
    file.close()
    

#WriteRaster("Potential_flow_length_2.asc", cc, xll = 1466225, yll = -3328900, cell_size = 25, NoData_value = -9999)


# read data

#### load the grid file
def LoadGrid(filepath):

    GridData = []
    grid_params = {}
    with open(filepath,'r') as fp:  
       for cnt, line in enumerate(fp):
           if line.startswith('ncols'):#cnt == 0:
               grid_params['ncols'] = int(line.split()[1])
           elif line.startswith('nrows'):#cnt == 1:
               grid_params['nrows'] = int(line.split()[1])
           elif line.startswith('xllcorner'):#cnt == 2:
               grid_params['xllcorner'] = float(line.split()[1])
           elif line.startswith('yllcorner'):#cnt == 3:
               grid_params['yllcorner'] = float(line.split()[1])
           elif line.startswith('cellsize'):#cnt == 4:
               grid_params['cellsize'] = float(line.split()[1])
           elif line.startswith('NODATA_value'):#cnt == 5:
               grid_params['NODATA_value'] = float(line.split()[1])
           else:
               GridData.append([float(i) for i in line.split()])
    
    data = np.zeros((grid_params['nrows'], grid_params['ncols']), dtype = float)
    
    count = -1
    for row in GridData:
        count += 1
        data[count] = row
        
    return data, grid_params

filepath = "Flooded_area_edit.asc"

def PlotRasters(raster):
    raster_inv = np.transpose(raster)
    raster_flip = np.flip(raster_inv,1)
    raster_value = raster_flip*25/1000
    
    nrows, ncols = np.shape(raster)
    x = np.arange(0,ncols*25/1000,25/1000)
    y = np.arange(0,nrows*25/1000,25/1000)
    Y, X = np.meshgrid(y,x)
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rc('font', size=18)
     
    im = ax.pcolormesh(X,Y, raster_value,shading='auto', cmap = 'RdBu_r')  
    
    # Create colorbar
    Legend = 'Potential connection length (km)'
    #Legend = 'Water level (m)'
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(Legend, rotation=-90, va="bottom")
    #fig.colorbar(im, ax=ax)
    
    ax.set_xlabel('X direction (km)')
    ax.set_ylabel('Y direction (km)')
    fig.tight_layout() 
    #plt.legend()
    plt.show()
    ax.tick_params(which='major', length=5)
    
    return fig


#%%%
# input the DEM, flow frequency and flow direction 

dem = rxr.open_rasterio('DEM_25m.tif')
max_flood = rxr.open_rasterio('Flooded_area_edit2.asc')
flow_dir = rxr.open_rasterio('Flow_direction.tif')

# convert into numpy arrays
dem = np.array(dem[0])
max_flood = np.array(max_flood[0])

#%%
## start iteration 
cc = np.zeros((dem.shape[0],dem.shape[1]))
save = np.zeros((dem.shape[0],dem.shape[1]))  

count = 0  
while count < 1000:
    count = count + 1
    
    # set output cell with 1
    cc[490:497,0] = 1
    
    # update cell with direction = 1
    save[flow_dir==1] = cc[flow_dir==1]+1
    save = np.roll(save,1)
    save[:,0] = 0
    # update flow value    
    cc[save>cc] = save[save>cc]        
    save = np.zeros((dem.shape[0],dem.shape[1]))
       
    # update cell with direction = 2
    save[flow_dir==2] = cc[flow_dir==2]+1
    save = np.roll(save,(1, 1), axis=(1, 0))
    save[:,0] = 0    
    save[0,:] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))

    # update cell with direction = 4
    save[flow_dir==4] = cc[flow_dir==4]+1
    save = np.roll(save,1, axis = 0)
    save[0,:] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))


    # update cell with direction = 8
    save[flow_dir==8] = cc[flow_dir==8]+1
    save = np.roll(save,(-1, 1), axis=(1, 0))
    save[:,-1] = 0
    save[0,:] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))

    # update cell with direction = 16
    
    save[flow_dir==16] = cc[flow_dir==16]+1
    save = np.roll(save,-1)
    save[:,-1] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))
        

    # update cell with direction = 32
    save[flow_dir==32] = cc[flow_dir==32]+1
    save = np.roll(save,(-1, -1), axis=(1, 0))
    save[:,-1] = 0
    save[-1,:] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))

    # update cell with direction = 64
    save[flow_dir==64] = cc[flow_dir==64]+1
    save = np.roll(save,-1,axis = 0)
    save[-1,:] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))
    
        # update cell with direction = 128
    save[flow_dir==128] = cc[flow_dir==128]+1
    save = np.roll(save,(1, -1), axis=(1, 0))
    save[-1,:] = 0
    save[:,0] = 0
    # update flow value
    cc[save>cc] = save[save>cc]
    save = np.zeros((dem.shape[0],dem.shape[1]))
    
    
#%%
# input: dem and maximun flooded area
# output: potential flow length

#assign all the downstream boundary ==1

cc = np.zeros((dem.shape[0],dem.shape[1]))
save = np.zeros((dem.shape[0],dem.shape[1]))  

cc[55:65,0] = 1

# calculate slopes in 4 direction
# slope1 = np.zeros((dem.shape(0),dem.shape(1)))
# slope2 = np.zeros((dem.shape(0),dem.shape(1)))
# slope3 = np.zeros((dem.shape(0),dem.shape(1)))
# slope4 = np.zeros((dem.shape(0),dem.shape(1)))

slope1 = dem - np.roll(dem,-1)
slope2 = dem - np.roll(dem,-1, axis = 0)
slope3 = dem - np.roll(dem,1)
slope4 = dem - np.roll(dem,1, axis = 0)

n = 0
# n < 100:  ### iteration
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


#%% compute the distribution of flow lengths
filepath = "D:\\CNN\\Trial_unet_FOM_WB_CL_2011\\CNN_273.asc"

filepath2 = "Z:\\work\\3_Flood_Inundation\\0_Working\\7_Chi\\NRRI_tests\\0_Data\Test\\Input\\Potential_flow_length_2.asc"

CNN_fm, grid_params = LoadGrid(filepath)

flow_len, grid_params = LoadGrid(filepath2)

flow_len_compare = np.where(CNN_fm>=0.5,flow_len,-9999) 

# calculate the flooded area

# calculate the distribution of the flow lengths => make a comparison

#%%
####### count frequency distribution of length of connection 
max_length = np.max(flow_len_compare)

###plot pr distribution
figure = plt.figure(figsize=(6, 5)) 
plt.rc('font', size=18)
#for i in range(len(connect_save)):
cc = np.array(flow_len_compare)
max_length = np.max(cc)
len_connect = np.arange(1,max_length+1,1)


mu = np.mean(raster[raster>0])  # mean of distribution
sigma = np.std(raster[raster>0])  # standard deviation of distribution

### plot distribution of length of flow connect
dist = np.zeros(len(len_connect))
prob = np.zeros(len(len_connect))
count = 0 

#value, counts = np.unique(raster, return_counts=True)
#prob = counts/sum(counts)

for k in len_connect:
    dist[count]  =  np.count_nonzero(cc==k)
    count += 1
#
prob = dist/sum(dist) #porbability distribution

#plt.plot(len_connect,prob, label = 't = ' + str(i*4*10) + 'hr')
plt.plot(len_connect*25/1000,prob, label = 't = 76 hrs')
plt.xlabel('Potential flow length (km)')
plt.ylabel('Relative frequency')
title  = '$\mu$ = ' + str(np.round(mu,1)) + ',' + '$\sigma$ = ' + str(np.round(sigma,1))
ax.set_title(title)

figure.tight_layout()    
plt.legend()
plt.show()


#%%
### Plot the probability distribution of the data
def ProDist(raster):
    
    fig, ax = plt.subplots()
    num_bins = 50
    n, bins, patches = ax.hist(raster[raster>0], num_bins, density=True)
    
    # add a 'best fit' line
    mu = np.mean(raster[raster>0])  # mean of distribution
    sigma = np.std(raster[raster>0])  # standard deviation of distribution
    
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Relative frequency')
    title  = '$\mu$ = ' + str(np.round(mu,1)) + ',' + '$\sigma$ = ' + str(np.round(sigma,1))
    ax.set_title(title)
    ax.tick_params(which='major', length=5)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
    return fig

fig = ProDist(flow_len_compare)


#%% function to plot and compare between different time steps in a flood eventfilepath2 = "Z:\\work\\3_Flood_Inundation\\0_Working\\7_Chi\\NRRI_tests\\0_Data\Test\\Input\\Potential_flow_length_2.asc"
flow_len, grid_params = LoadGrid(filepath2)

timesteps = np.array([10, 33, 85, 117, 177, 250])


# read the flood extent file 
ts = int(25)

#filepath = "D:\\CNN\\Trial_unet_FOM_WB_BC_2011\\CNN_" + str(ts).zfill(3)+ ".asc"


filepath = "D:\\CNN\\Trial_unet_fre_CL_fullts\\CNN_" + str(ts).zfill(3)+ ".asc"
CNN_fm, grid_params = LoadGrid(filepath)

def Plot_relative_frequency(CNN_fm, flow_len):

    # assign the flow length associate with with the flood map, ignore NoData values
    cc = np.where(CNN_fm>=0.5,flow_len,-9999) 
    
    # calculate the probability distribution:
    num_bins = 50
    
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


fm_len = np.where(CNN_fm>=0.5,flow_len,0) 

fig = Plot_relative_frequency(CNN_fm, flow_len)
fig1 = PlotRasters(fm_len)


#%% plot the results

timestep = np.array([25,49,52,101,108,116])

for ts in timestep:
    
    ts = int(ts)

    filepath = "D:\\CNN\\Trial_unet_fre_CL_fullts\\CNN_" + str(ts).zfill(3)+ ".asc"
    CNN_fm, grid_params = LoadGrid(filepath)
    
    # plot the data
    fm_len = np.where(CNN_fm>=0.5,flow_len,0) 
    fig = Plot_relative_frequency(CNN_fm, flow_len)
    fig1 = PlotRasters(fm_len)
    


fig1 = PlotRasters(flow_len)









