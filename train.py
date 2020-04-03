# This code contains the necessary to load the old model, add new layers and train the new layers from scratch with the available data. More code wrt preprocessing may be added later.
# This code might be pushed into a class, so necessary precaution must be taken


# import necessary libraries

# deeplearning
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D, GlobalMaxPool2D
from models.mobilenet_model import get_mobilenet_model


# plotting and math
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math
from numpy import ma
from scipy.ndimage.filters import gaussian_filter


# Load existing model
model = get_mobilenet_model(1.0, 224)

# Load and add weights to the model
#weights_path = "weights.best.mobilenet.h5" # weights tarined from scratch 
#model.load_weights(weights_path)

#model.save('old_mobilenet_model.h5')
#model.summary()

new_model = load_model('old_mobilenet_model.h5')

# freeze older layers
for layer in new_model.layers:
    layer.trainable = False

# start adding new layers
out = new_model.output
out = tf.concat(axis=3, values=[out[2], out[3]], name='concat_stage2_outs')
out = MaxPool2D(pool_size=(2,2), strides=None, padding='same')(out)
#out = Conv2D(64,kernel_size=(3,3),padding='same', activation='relu')(out)
#out = MaxPool2D(pool_size=(2,2), strides=None, padding='same')(out)
out = Conv2D(128,kernel_size=(3,3),padding='same', activation='relu')(out)
out = MaxPool2D(pool_size=(2,2), strides=None, padding='same')(out)
out = GlobalMaxPool2D()(out)

#out = Flatten()(out)

# block for left hand
#out_left = Dense(1024, activation = 'relu')(out)
#out_left = Dropout(0.5)(out_left)
out_left = Dense(512, activation = 'relu')(out)
out_left = Dropout(0.5)(out_left)
out_left = Dense(256, activation = 'relu')(out_left)
out_left = Dropout(0.5)(out_left)
out_left = Dense(22, activation = 'softmax', name = 'left_out')(out_left)

# block for right hand
#out_right = Dense(1024, activation = 'relu')(out)
#out_right = Dropout(0.5)(out_right)
out_right = Dense(512, activation = 'relu')(out)
out_right = Dropout(0.5)(out_right)
out_right = Dense(256, activation = 'relu')(out_right)
out_right = Dropout(0.5)(out_right)
out_right = Dense(22, activation = 'softmax', name='right_out')(out_right)

# concatenate left and right blocks as outputs for final output
test_model = keras.Model(inputs = new_model.input, outputs = [out_left,out_right])
test_model.summary()


