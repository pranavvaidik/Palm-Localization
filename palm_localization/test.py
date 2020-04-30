from config import palm_localization_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor
from tools.io import HDF5DatasetWriter
from tools.datasets import MultiOutputDatasetLoader
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
import pickle

# grab the paths to the images
imagePaths_left = list(paths.list_images('../data/left/'))
imagePaths_right = list(paths.list_images('../data/right/'))

labels_left = []
labels_right = []

# loop over all image paths
print(len(imagePaths_right), len(imagePaths_left))

for i, imagePath_right in enumerate(imagePaths_right):
	# get the image name
	image_name = imagePath_right.split(os.path.sep)[-1]

	# get the image path for right hand
	imagePath_left = [path for path in imagePaths_left if path.split(os.path.sep)[-1] == image_name]
	
	if len(imagePath_left) == 0:
		print ("check this path: ", imagePath_right)
	elif len(imagePath_left)>1:
		print("duplicates exist")
		print("original file is:", imagePath_right)
		print("duplicates on right:", imagePath_left)
	


