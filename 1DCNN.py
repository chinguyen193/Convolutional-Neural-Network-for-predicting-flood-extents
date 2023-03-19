# -*- coding: utf-8 -*-
"""
This is the code to run the 1D CNN models

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
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, BatchNormalization, Dropout, Activation, Lambda
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



print(tf.__version__)

#%% Custom loss function
def custom_loss(y_true, y_pred):
    
    correct_wet = tf.math.reduce_sum(y_true*y_pred)
    obs_wet = tf.math.reduce_sum(y_true)
    wrong_wet = tf.math.reduce_sum((1-y_true)*y_pred) 
    CSI = correct_wet/(obs_wet + wrong_wet)  
    loss = 1 - CSI

    return loss

#%%  Read the input file (form of a library) for the input training data from the WB station
# The structure of the data was introduce in the "Prepare_inputs.py" file

import pickle
# Narran Park data
filepath = 'workspace_1D_WB.pickle'  

# Wilby data
#filepath = 'workspace_1D_WB.pickle'

with open(filepath, 'rb') as f:
    Data = pickle.load(f)
    
x_train, Y_train, x_test, Y_test, steps, features, outputs, X_pred = Data['x_train'], Data['Y_train'], Data['x_test'], Data['Y_test'], Data['steps'], Data['features'], Data['outputs'], Data['X_pred']

del Data



#%% 1D CNN model

def CNN1D_Model_newloss(x_train, Y_train, x_test, Y_test, steps, features, outputs):
    '''
    Two layered conv network
    '''
    print('Running the CNN model...')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=1, input_shape=(steps, features)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(128, kernel_size=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, kernel_size=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    model.add(Flatten())
    model.add(Dense(32, kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512,kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(outputs))
    model.add(Activation('sigmoid'))
    
    #model.add(Lambda(func))
    #model.add(Activation(custom_activation, name='SpecialActivation'))
    # create a Keras function to get i-th layer    
    optimizer = Adamax(lr=0.001)  #try learning rate 0.001, 
    
    #define loss function
    #model.compile(loss=custom_loss, metrics=['binary_accuracy'], optimizer= optimizer)
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer= optimizer)
    print(model.summary())
    
    ##start time
    start = timeit.default_timer()
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=1, mode='auto')
    history = model.fit(x_train,Y_train,validation_data=(x_test,Y_test),batch_size=32,callbacks=[monitor],verbose=1,epochs=200)
    
    # evaluate the model
    _, train_acc = model.evaluate(x_train, Y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, Y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    # no early stopping
    #history = model.fit(x_train,Y_train,validation_data=(x_test,Y_test),batch_size=32,verbose=0,epochs=100) #nb of epochs = 100
    
    print(history.history.keys())
    
    #stop time
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    # plot history
    plt.figure(1) 
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    #plot accuracy 
    plt.figure(2)
    plt.plot(history.history['binary_accuracy'], label='train')
    plt.plot(history.history['val_binary_accuracy'], label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    
    return model


#%% save and load model for prediction

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

#%% run the prediction

def predict(model,X_Test):
  
  start = timeit.default_timer()
    
  tar_dir = '/fs03/yz84/Chi_CNN/CNN1D_test/' # change this directory for targets according to model..directory for The CNN output files
  pathlib.Path(tar_dir).mkdir(parents=True, exist_ok=True)
  data = rio.open('Initial.asc') #reference image for fixing raster dimensions
   
  ##Make predictions
  for i in [58, 90, 113, 138, 145, 170, 218, 225, 250, 266, 273, 282, 298]:
    x_test = X_Test[i]
    x_test = x_test.reshape(1,1,X_Test.shape[1])
    y_pred = model.predict(x_test)
    y_pred.resize(data.height, data.width)
    
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>=0.5] = 1
    
    
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


#%% Run the simulation  

model = CNN1D_Model_newloss(x_train, Y_train, x_test, Y_test, steps, features, outputs)


#%%
path = '/fs03/yz84/Chi_CNN'

#save_model(model, name='CNN1D_CL1_trial3_NP')

load_model(name='CNN1D_test')

model = load_model(name ='CNN1D_test')

predict(model, X_pred)
