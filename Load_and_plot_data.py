# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:08:56 2021

@author: tpngu28


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import rc
from matplotlib import cm
import os

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

#### Plot raster layer

def PlotRasters(raster, Legend):
    raster_inv = np.transpose(raster)
    raster_flip = np.flip(raster_inv,1)
    nrows, ncols = np.shape(raster)
    x = np.arange(0,ncols,1)
    y = np.arange(0,nrows,1)
    Y, X = np.meshgrid(y,x)
    fig, ax = plt.subplots(figsize=(4.9, 4))
    
    
    im = ax.pcolormesh(X,Y, raster_flip,shading='auto', cmap = 'RdBu_r')    
    
    # Create colorbar
    #Legend = 'Connectivity length (pixels)'
    #Legend = 'Water level (m)'
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(Legend, rotation=-90, va="bottom")
    #fig.colorbar(im, ax=ax)
    
    ax.set_xlabel('X direction (pixels)')
    ax.set_ylabel('Y direction (pixels)')
    
    plt.rc('font', size=20)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.tight_layout() 
    plt.legend()
    plt.show()
    
    return fig
