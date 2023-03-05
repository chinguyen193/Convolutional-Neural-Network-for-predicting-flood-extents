# -*- coding: utf-8 -*-
"""
Custom Loss function for the CNN model to train the flood extents
The custom loss function is based on the CSI index.
"""


def custom_loss(y_true, y_pred):
    
    correct_wet = tf.math.reduce_sum(y_true*y_pred)
    obs_wet = tf.math.reduce_sum(y_true)
    wrong_wet = tf.math.reduce_sum((1-y_true)*y_pred) 
    CSI = correct_wet/(obs_wet + wrong_wet)
    loss = 1 - CSI
    
    return loss

