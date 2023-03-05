# -*- coding: utf-8 -*-
"""
This is the script to prepare input data to train 1D and 2D CNN models
- Input training data for the 1D CNN model is the input discharges at Narran Park or at Wilby Wilby
- The training data for he 2D CNN model is the input discharges at Narran Park or at Wilby Wilby and the 2D feature layers including 
the Flood Occurance Map (FOM), the Digital Elevation Model (DEM), the slope, the curvature ang the aspect. 

"""
# Load all libraries and software packages

import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import os
import urllib
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr 
import scipy.ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Load_and_plot_data import LoadGrid
from Raster_feature import feature_fast_mean

#%% Read the discharge or water level data from csv file

def read_data(filename):
    file = pd.read_csv(filename ,skiprows = 10, names=['time','value', 'quality', 'type'])
    file['time'] = pd.to_datetime(file['time'], format='%Y-%m-%dT%H:%M:%S.000+10:00') 

    file.index = file['time']
    df = file.resample('d').mean()  #mean() #cannot use sum() because in some year there is only one record value per day
    time = df.index
    Q = df['value']
    
    return df, time, Q

#%% Function to build training data for the 1D model

################# prepare X-train and X-test 

def X_data(ds, Q_Wilby, num_backts=61):
    Time_st = pd.to_datetime(ds.time.values, format='%Y-%m-%dT%H:%M:%S.000000000')
    Q_st = np.zeros((Time_st.shape[0], num_backts+1))

    #calculate the start day of the flood event
    idx0 = Q_Wilby.index.get_loc(Time_st[0], method='nearest')
    time0 = Q_Wilby.index[idx0]
    
    for i in range(int(Time_st.shape[0])):
        idx = Q_Wilby.index.get_loc(Time_st[i], method='nearest')
        
        #find the closest Q to the timestep of the satellite image
        Q_save = np.array(Q_Wilby.loc[Q_Wilby.index[idx-num_backts : idx],'value']) 
        Q_st[i][0] = (Q_Wilby.index[idx] - time0).days
        Q_st[i][1:] = np.array(Q_save[::-1])
        
        # change nan to 0 
        Q_st = np.nan_to_num(Q_st)
        
    return Q_st

################## prepare Y-train and Y-test 

def Y_data(ds):
    Fmap_flat = np.zeros((ds.water.shape[0],ds.water.shape[1]*ds.water.shape[2]))
    Time_st = pd.to_datetime(ds.time.values, format='%Y-%m-%dT%H:%M:%S.000000000')
    for i in range(int(Time_st.shape[0])):
        
        # for some reason, wet pixel was saved as 'nan', NEED TO CHECK AGAIN!!!
        Fmap = np.nan_to_num(ds.water.values[i],nan = 1)  
        Fmap_flat[i][:] = Fmap.reshape(1,Fmap.shape[0]*Fmap.shape[1])
    
    return Fmap_flat


#%% Function to build training data for the 2D model

################# prepare X-train and X-test 

def X_data_2D(ds, Q_Wilby, num_backts=61):
    Time_st = pd.to_datetime(ds.time.values, format='%Y-%m-%dT%H:%M:%S.000000000')
    Q_st = np.zeros((Time_st.shape[0], num_backts+1))

    #calculate the start day of the flood event
    idx0 = Q_Wilby.index.get_loc(Time_st[0], method='ffill')
    time0 = Q_Wilby.index[idx0]
    
    delta_T = np.zeros((Time_st.shape[0],1))
    for i in range(int(Time_st.shape[0])):
        idx = Q_Wilby.index.get_loc(Time_st[i], method='ffill')
        
        #find the closest Q to the timestep of the satellite image
        Q_save = np.array(Q_Wilby.loc[Q_Wilby.index[idx-num_backts+1 : idx+1],'value']) 
        Q_st[i][0] = (Q_Wilby.index[idx] - time0).days
        Q_st[i][1:] = np.array(Q_save[::-1])
        
        # change nan to 0 
        Q_st = np.nan_to_num(Q_st)
        
        if i > 0:
            delta_T[i] =  Q_st[i][0] - Q_st[i-1][0]
        
    return Q_st, delta_T


################## prepare Y-train and Y-test 

def Y_data_2D(ds):
    
    shape = int(512)
    Fmap_2D = np.zeros((ds.water.shape[0],shape,shape))
    Time_st = pd.to_datetime(ds.time.values, format='%Y-%m-%dT%H:%M:%S.000000000')
    for i in range(int(Time_st.shape[0])):
        
        # for some reason, wet pixel was saved as 'nan', NEED TO CHECK AGAIN!!!
        Fmap = np.nan_to_num(ds.water.values[i],nan = 1)  
        Fmap_2D[i,2:,0:Fmap.shape[1]] = Fmap.reshape(1,Fmap.shape[0],Fmap.shape[1])     
            
    
    return Fmap_2D
#%% Function to build X data for prediction 

################# prepare X-test for prediction
def X_prediction(Q_Wilby, interval =1 ,num_backts = 61):
    # start_date = pd.to_datetime('10-08-2010 00:00:00') #M-D-Y
    # end_date = pd.to_datetime('10-21-2011 00:00:00') #M-D-Y
    start_date = pd.to_datetime('11-20-2011 00:00:00')  #M-D-YYYY
    end_date = pd.to_datetime('09-23-2012 00:00:00')
    
    
    idx0 = Q_Wilby.index.get_loc(start_date, method='nearest')
    
    
    #prepare the X_pred file:        
    num_timestep = int(np.rint((end_date-start_date).days/interval))
    X_pred = np.zeros((num_timestep,num_backts+1))
    for i in range (num_timestep):
        X_pred[i][0] = i*interval
        idx = idx0 + i*interval
        Q_save = Q_Wilby['value'].iloc[idx-num_backts:idx].values
        X_pred[i][1:] = np.array(Q_save[::-1])
        X_pred = np.nan_to_num(X_pred)
    return X_pred



#%% Read input discharge data and feature data

# Read discharge data 
filename = 'Q_NP.csv'  # Read discharge data at Narran Park
Q, time, Q_value = read_data(filename)

# Read satellite dataset (from the WOfS and the Sentinel-2 images)

###### for the NP discharge data

ls = ['Flood14.nc','Flood15n.nc','Flood16n.nc','Flood17n.nc','Flood18n.nc','Flood19.nc','Flood20.nc','Flood21_comb.nc',
      'Flood22_comb.nc','Flood23_combn.nc']

###### for the WB discharge data
# ls = ['Flood1.nc','Flood2n.nc','Flood3-4n.nc','Flood5.nc','Flood6n.nc','Flood7n.nc',
#       'Flood8n.nc','Flood9.nc','Flood10n.nc','Flood11.nc','Flood12n.nc','Flood13n.nc','Flood14.nc',
#       'Flood15n.nc','Flood16n.nc','Flood17n.nc','Flood18n.nc','Flood19.nc','Flood20.nc','Flood21_comb.nc',
#       'Flood22_comb.nc','Flood23_combn.nc']


# Read DEM data
filepath = "DEM.asc"  # original DEM with no-data pixels
DEM_org, grid_params = LoadGrid(filepath)

# pad the DEM data to the size of (512,512). The size of the input data is chosen so that it is easier to train the CNN model. 
DEM_edit = np.zeros((512,512)) + -9999
DEM_edit[2:,0:DEM_org.shape[1]] = np.array(DEM_org) 

# create mask layer
mask = np.zeros((512,512))
mask[DEM_edit>0] = 1

# replace no-data pixels in the DEM with max elevation 
DEM = np.array(DEM_edit)
DEM[DEM<0] = np.amax(DEM_edit)

# create feature layers
slop,curvature,cos,sin,aspect = feature_fast_mean(DEM_edit, group_size=3, cell_size=25)


#%% Main code for prepare data for 1D CNN model
###################################################

num_backts = 61 #number of back time steps
X_org =  np.empty((0,num_backts+1))
Y = np.empty((0,226440))

for i in ls:    
    ds_flood = xr.open_dataset(i)
    # X dataset
    Q_st = X_data(ds_flood, Q, num_backts)
    X_org = np.append(X_org, Q_st, 0)
    # Y dataset
    Fmap_flat = Y_data(ds_flood)
    Y = np.append(Y, Fmap_flat, 0)

X1_org = X_org[:,0].reshape((len(X_org), 1)) # time step t
X2_org = X_org[:,1:] # Discharge at time t to t-61

### prediction  data

X_pred_org = X_prediction(Q, interval = 1 ,num_backts = 61)
X1_pred_org = X_pred_org[:,0].reshape((len(X_pred_org), 1)) # time step t
X2_pred_org = X_pred_org[:,1:] # Discharge at time t to t-61

# Select X_train and X_test
scaler = MinMaxScaler()  # Scale the data into the range [0,1]

X1 = scaler.fit_transform(X1_org)
X2 = scaler.fit_transform(X2_org.reshape(-1, 1)).reshape(X2_org.shape[0],X2_org.shape[1])

X1_pred  = np.amax(X1_pred_org)/np.amax(X1_org)*scaler.fit_transform(X1_pred_org)
X2_pred  = np.amax(X2_pred_org)/np.amax(X2_org)*scaler.fit_transform(X2_pred_org.reshape(-1, 1)).reshape(X2_pred_org.shape[0],X2_pred_org.shape[1])

X = np.concatenate((X1,X2),axis=1)  
X_pred = np.concatenate((X1_pred,X2_pred),axis=1)


####### Select X_train and X_test for Q at Narran Park
X_train = X[56:]
X_test =  X[0:55]

#Select Y_train and Y_test
Y_train = Y[56:]
Y_test =  Y[0:55]

# ####### Select X_train and X_test for Q at Wilby Wilby
# X_train = X[127:]
# X_test =  X[0:127]

# #Select Y_train and Y_test
# Y_train = Y[127:]
# Y_test =  Y[0:127]

# reshape x_train and x_test
x_train= X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
x_test= X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

### parameters
steps = x_train.shape[1]
features = x_train.shape[2]
outputs = Y_train.shape[1]

# test data input
print('X Train shape:', x_train.shape, 'X Test shape:', x_test.shape, 'Train target shape:',Y_train.shape, 'Test target shape:',Y_test.shape)

# save data into library file 
import pickle
filepath = 'workspace_1D_NP.pickle'

Data = {'X1': X1, 'X2': X2, 'Y': Y, 'X1_pred': X1_pred, 'X2_pred': X2_pred}

with open(filepath, 'wb') as f:
    pickle.dump(Data, f)    


#%% Main code for prepare data for 2D CNN model
###################################################

num_backts = 61 #number of back time steps
X =  np.empty((0,num_backts+1))
Y = np.empty((0,512,512)) # height = 510, width = 510
X_dt =  np.empty((0,1))

for i in ls:    
    ds_flood = xr.open_dataset(i)
    # X dataset
    Q_st, delta_T = X_data_2D(ds_flood, Q, num_backts)
    X = np.append(X, Q_st, 0)
    X_dt = np.append(X_dt,delta_T, 0) # delta timestep from the previous timestep
    # Y dataset
    Fmap_2D = Y_data_2D(ds_flood)
    Y = np.append(Y, Fmap_2D, 0)

X1_org = X[:,0].reshape((len(X), 1)) # time step t
X2_org = X[:,1:] # Discharge at time t to t-61


####

# scale the data in range [0,1]

scaler = MinMaxScaler()
scaler1 = StandardScaler()

layer_DEM = scaler.fit_transform(DEM.reshape(-1, 1)).reshape(512,512)

DEM_train = np.zeros((440,512,512))

for i in range(0,440):
    DEM_train[i,:,:] =np.array(layer_DEM)
    
DEM_test = np.zeros((126,512,512))

for i in range(0,126):
    DEM_test[i,:,:] =np.array(layer_DEM)


scaler = MinMaxScaler()
X1 = scaler.fit_transform(X1_org)
X2 = scaler.fit_transform(X2_org.reshape(-1, 1)).reshape(X2_org.shape[0],X2_org.shape[1])
 

X1_train = X1[127:]
X1_test =  X1[0:126]

X2_train = X2[127:]
X2_test =  X2[0:126]

X_dt_train = X_dt[127:]
X_dt_test = X_dt[0:126]

#Select Y_train and Y_test
Y_train = Y[127:]
Y_test =  Y[0:126]

x1_train= X1_train.reshape(X1_train.shape[0], 1, X1_train.shape[1])
x1_test= X1_test.reshape(X1_test.shape[0], 1, X1_test.shape[1])

x2_train= X2_train.reshape(X2_train.shape[0], 1, X2_train.shape[1])
x2_test= X2_test.reshape(X2_test.shape[0], 1, X2_test.shape[1])

steps = x1_train.shape[1]
features1 = x1_train.shape[2]
features2 = x2_train.shape[2]
outputs = Y_train.shape[1]

print('X1 Train shape:', X1.shape, 'X Test shape:', x1_test.shape, 'Train target shape:',Y_train.shape, 'Test target shape:',Y_test.shape)

# prepare input data for model simulation

X_pred = X_prediction(Q, interval =1 ,num_backts = 61)

X1_pred_org = X_pred[:,0].reshape((len(X_pred), 1)) # time step t
X2_pred_org = X_pred[:,1:] # Discharge at time t to t-61

DEM_pred = np.zeros((len(X_pred),512,512))

for i in range(0,len(X_pred)):
    DEM_pred[i,:,:] =np.array(layer_DEM)

#normalize the data
X1_pred  = scaler.fit_transform(X1_pred_org)
X2_pred  = scaler.fit_transform(X2_pred_org.reshape(-1, 1)).reshape(X2_pred_org.shape[0],X2_pred_org.shape[1])


#save_data

import pickle
filepath = 'workspace_2D.pickle'


Data = {'x1_train':x1_train,'x2_train':x2_train,'DEM_train': DEM_train, 'DEM_test':DEM_test,'Y_train': Y_train, 'x1_test' : x1_test,'x2_test': x2_test , 'Y_test':Y_test, 'steps':steps, 'X1_pred':X1_pred, 'X2_pred':X2_pred, 'DEM_pred':DEM_pred}

with open(filepath, 'wb') as f:
    pickle.dump(Data, f)
    
    

























