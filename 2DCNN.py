# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:43:50 2023

@author: ngu204
"""

#%%
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, BatchNormalization, Dropout, Activation, Lambda, RepeatVector, Reshape
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.activations import tanh
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils.generic_utils import get_custom_objects
import pandas as pd
import numpy as np
import timeit
import os
import rasterio as rio
import pathlib
import xarray as xr
import pickle

from keras.layers import *
from keras.regularizers import l2 

print(tf.__version__)

#%% Custom loss function

def custom_loss(y_true, y_pred):
    
    correct_wet = tf.math.reduce_sum(y_true*y_pred)
    obs_wet = tf.math.reduce_sum(y_true)
    wrong_wet = tf.math.reduce_sum((1-y_true)*y_pred) 
    CSI = correct_wet/(obs_wet + wrong_wet)
    loss = 1 - CSI
    return loss

#%% import 2D input data

filepath = 'workspace_2Dn.pickle'
with open(filepath, 'rb') as f:
    Data = pickle.load(f)
    
x1_train, x2_train, DEM_train, DEM_test, Y_train, x1_test, x2_test , Y_test, steps, X1_pred, X2_pred, DEM_pred = Data['x1_train'], Data['x2_train'], Data['DEM_train'], Data['DEM_test'], Data['Y_train'], Data['x1_test'], Data['x2_test'], Data['Y_test'], Data['steps'], Data['X1_pred'], Data['X2_pred'], Data['DEM_pred']

del Data

#%% 

# import the flood frequency map

filename = 'Wofs_frequency.nc'
ds = xr.open_dataset(filename)

# reshape and remove nan values
frequency = ds.frequency.values
frequency = frequency.reshape((510,444,1))
frequency = np.nan_to_num(frequency)
fre_array = np.zeros((512,512,1))
fre_array[2:,0:frequency.shape[1],:] = np.array(frequency)

# Scale data into the range [0,1]
scaler = MinMaxScaler()

fre_array = scaler.fit_transform(fre_array.reshape(-1, 1)).reshape((512,512,1))

#%% reshape the dimentions of the input data

#use flood occurrence map as feature layer
dem_array = np.array(fre_array)

#train
X1_patch_train = np.array(x1_train).reshape((x1_train.shape[0],1))
X2_patch_train = np.array(x2_train).reshape((x2_train.shape[0],61))
#DEM_patch_train = np.array(DEM_train).reshape((DEM_train.shape[0],DEM_train.shape[1],DEM_train.shape[2],1))
DEM_patch_train = np.repeat(fre_array.reshape((1,512,512,1)), X1_patch_train.shape[0], axis = 0)
Y_patch_train = np.array(Y_train)

#test
X1_patch_test = np.array(x1_test).reshape((x1_test.shape[0],1))
X2_patch_test = np.array(x2_test).reshape((x2_test.shape[0],61))
#DEM_patch_test = np.array(DEM_test).reshape((DEM_test.shape[0],DEM_test.shape[1],DEM_test.shape[2],1))
DEM_patch_test = np.repeat(fre_array.reshape((1,512,512,1)), X1_patch_test.shape[0], axis = 0)
Y_patch_test = np.array(Y_test)

#predict
X1_pred = np.array(X1_pred).reshape((X1_pred.shape[0],1))
X2_pred = np.array(X2_pred).reshape((X2_pred.shape[0],61))

# check the dimentions of data arrays
print(Y_patch_train.shape)
print(X2_patch_train.shape)
print(DEM_patch_train.shape)

#%%

def CNN_DEM_unet(X1_patch_train, X2_patch_train, DEM_patch_train, Y_patch_train, X1_patch_test, X2_patch_test, DEM_patch_test, Y_patch_test, patch_size = 512, steps = 1):
    
    print('Running the CNN model...')
    
    
    filtersize = 3 ### set filter size = 3
    num_layer = 4 ### nb of convolutional layers
    
    # Reshape 1D input discharge
    L_shape = int(patch_size/(np.power(2,num_layer-1))) #shape of last convolution layer
    
    
    Input_ts = Input(shape=(X1_patch_train.shape[1])) #( , 1)
    #Ts = Dense(1,activation='relu')(Input_ts)
    Ts = RepeatVector(L_shape*L_shape*30,)(Input_ts) 
    Ts = Reshape((L_shape, L_shape, 30))(Ts)   # (None, 16, 16, 20)
        
    Input_Q = Input(shape=(X2_patch_train.shape[1])) #( , 61)
    #Q = Dense(30,activation='relu')(Input_Q)
    Q = RepeatVector(L_shape*L_shape)(Input_Q) 
    Q = Reshape((L_shape, L_shape, 61))(Q) #(None, 16, 16, 61)
    
    Input_t_Q = concatenate([Ts,Q])
    Input_t_Q = Conv2D(91, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(Input_t_Q)
    
    
    # Input DEM 
    
    ##### training model Unet
    Input_DEM = Input(shape=(patch_size, patch_size,1))
    #DEM = Reshape((patch_size, patch_size, 1))(Input_DEM)
    
    #layer1
    conv1 = Conv2D(16, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(Input_DEM)
    conv1 = Conv2D(16, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #layer2
    conv2 = Conv2D(32, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(pool1)
    conv2 = Conv2D(32, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #layer3
    conv3 = Conv2D(64, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(pool2)
    conv3 = Conv2D(64, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv3)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #layer4
    conv4 = Conv2D(128, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(pool3)
    conv4 = Conv2D(128, filtersize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv4)
    conv4 = BatchNormalization(momentum=0.9)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv5)
    conv5 = BatchNormalization(momentum=0.9)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(UpSampling2D(size = (2,2))(conv5)) # (None, 16, 16, 128)
    merge6 = concatenate([conv4, up6, Input_t_Q], axis = 3) # (None, 16, 16, 262)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv6)
    conv6 = BatchNormalization(momentum=0.9)(conv6)
    conv6 = Dropout(0.2)(conv6)
    
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3) # (None, 32, 32, 128)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv7)
    conv7 = BatchNormalization(momentum=0.9)(conv7)
    conv7 = Dropout(0.2)(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3) # (None, 64, 64, 64)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv8)
    conv8 = BatchNormalization(momentum=0.9)(conv8)
    conv8 = Dropout(0.2)(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3) # (None, 128, 128, 32)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(merge9)
    conv9 = Conv2D(16, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer ='l2')(conv9) # (None, 128, 128, 1) 
    conv9 = BatchNormalization(momentum=0.9)(conv9)
    conv9 = Dropout(0.2)(conv9)
      
    
    output = Dense(16,activation='relu')(conv9)
    output = Dense(1,activation='sigmoid')(output)
    
    model = Model(inputs=[Input_ts, Input_Q, Input_DEM], outputs=output) 
    
    #####
    
    optimizer = Adam(lr=0.0001)  #try learning rate 0.001, 
    
    #define loss function
    model.compile(loss=custom_loss, metrics=['binary_accuracy'], optimizer= optimizer)
    
    print(model.summary())
    
    ##start time
    start = timeit.default_timer()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
    history = model.fit([X1_patch_train, X2_patch_train, DEM_patch_train],Y_patch_train,validation_data=([X1_patch_test, X2_patch_test, DEM_patch_test],Y_patch_test),batch_size=32,callbacks=[monitor],verbose=1,epochs=200)
    #history = model.fit([X1_patch_train, X2_patch_train, DEM_patch_train],Y_patch_train,validation_data=([X1_patch_test, X2_patch_test, DEM_patch_test],Y_patch_test),batch_size=3,verbose=1,epochs=200)
    # evaluate the model
    _, train_acc = model.evaluate([X1_patch_train, X2_patch_train, DEM_patch_train], Y_patch_train, verbose=0)
    _, test_acc = model.evaluate([X1_patch_test, X2_patch_test, DEM_patch_test], Y_patch_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        
    print(history.history.keys())
    
    #stop time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    # plot history
    plt.figure() 
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    #plot accuracy 
    plt.figure()
    plt.plot(history.history['binary_accuracy'], label='train')
    plt.plot(history.history['val_binary_accuracy'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    
    return model

#%% run the model training

model = CNN_DEM_unet(X1_patch_train, X2_patch_train, DEM_patch_train, Y_patch_train, X1_patch_test, X2_patch_test, DEM_patch_test, Y_patch_test, patch_size = 512, steps = 1)


#%%

def save_model(model, name):
  # Save the weights
  model.save_weights(name+'.h5')

  # Save the model architecture
  with open(name+'.json', 'w') as f:
    f.write(model.to_json())

def load_model(name):
  ##Loading the model weights
  from tensorflow.keras.models import model_from_json

  # Model reconstruction from JSON file
  with open(name+'.json', 'r') as f:
      model = model_from_json(f.read())

  # Load weights into the new model
  model.load_weights(name+'.h5')

  return model


save_model(model, name='Trial_unet_fre_CL')

#%%
def predict(model, X1_pred, X2_pred, dem_array, patch_size = 512):
    
    start = timeit.default_timer()
    
    tar_dir = '/fs03/yz84/Chi_CNN/CNN_DEM/Trial_unet_fre_CL/' # change this directory for targets according to model..directory for The CNN output files
    pathlib.Path(tar_dir).mkdir(parents=True, exist_ok=True)
    data = rio.open('Initial.asc') #reference image for fixing raster dimensions
   
  ### Make predictions
  
    for i in [58, 90, 113, 138, 145, 170, 218, 225, 250, 266, 273, 282, 298]:

    ### Split terrain file to input to the model
        DEM_patch = np.array(dem_array).reshape((1,512,512,1))
        x1_pred = np.array(X1_pred[i]).reshape((1,X1_pred.shape[1]))
        x2_pred = np.array(X2_pred[i]).reshape((1,X2_pred.shape[1]))
            
        # model prediction    
        y_pred_patch = model.predict([x1_pred,x2_pred,DEM_patch])
        y_pred_patch = y_pred_patch.reshape((patch_size,patch_size))
        
        
        y_pred = np.array(y_pred_patch[2:,0:444])
    
        #y_pred[y_pred<0.5] = 0
        #y_pred[y_pred>=0.5] = 1
        
        src = data
        with rio.Env():
            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = src.profile
    
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(dtype=str(y_pred.dtype), count=1,compress='lzw')
    
            with rio.open(tar_dir+'CNN_{:03}'".asc".format(i), 'w', **profile) as dst:
            #with rio.open(tar_dir+fname+index+'.tif', 'w', **profile) as dst:
                dst.write(y_pred, 1)
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)   
    
    
#%%

def load_model(name):
  ##Loading the model weights
  from tensorflow.keras.models import model_from_json

  # Model reconstruction from JSON file
  with open(name+'.json', 'r') as f:
      model = model_from_json(f.read())

  # Load weights into the new model
  model.load_weights(name+'.h5')

  return model

model = load_model(name ='Trial_unet_fre_CL')

predict(model, X1_pred, X2_pred, dem_array, patch_size = 512)
