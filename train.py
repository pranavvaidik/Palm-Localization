# import necessary libraries

# deeplearning
import cv2
import tensorflow as tf
from palm_net import PalmNet
from tensorflow.keras.optimizers import SGD

# preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing_tools import MultiOutputDatasetLoader, ImageToArrayPreprocessor, SimplePreprocessor

# plotting and math
from matplotlib import pyplot as plt
import numpy as np
from imutils import paths
import argparse


# load image list for both left and right hands
print("[INFO] loading images.....")
imagePaths_left = list(paths.list_images('data/left'))
imagePaths_right = list(paths.list_images('data/right'))

# initialize image processors
sp = SimplePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

# load dataset and scale the pixels to range [0,1]
mdl = MultiOutputDatasetLoader(preprocessors=[sp,iap])
(data, labels_left, labels_right) = mdl.load(imagePaths_left, imagePaths_right, verbose = 500)



