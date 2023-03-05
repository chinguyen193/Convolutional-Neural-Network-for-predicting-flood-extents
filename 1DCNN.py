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

#%%
