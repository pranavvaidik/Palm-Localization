# import packages

# set the matplotlib backend so figures can be saved in background
import matplotlib
matplotlib.use("Agg")

# other imports
from config import palm_localization_config as config
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
# there should be an import from callbacks, add it
from tools.io import HDF5DatasetGenerator
from tools.nn.conv import PalmNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os

# construct training image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.1,
			height_shift_range = 0.1, shear_range = 0.15, horizontal_flip = False, 
			fill_mode = "nearest")

# load the RGB means for training set
means =  json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(224,224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()



