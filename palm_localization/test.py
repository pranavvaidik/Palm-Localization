from config import palm_localization_config as config
from tools.datasets import MultiOutputDatasetLoader
from imutils import paths
import numpy as np
import json
import cv2
import os
import pickle

image_folder_path = {'left' : '../data/left/', 'right' : '../data/right/'}

# grab the paths to the images
imagePaths_left = list(paths.list_images('../data/left/'))
imagePaths_right = list(paths.list_images('../data/right/'))

labels_left = []
labels_right = []

# loop over all image paths
print(len(imagePaths_right), len(imagePaths_left))

mdl = MultiOutputDatasetLoader()
trainLabels_left, trainLabels_right = mdl.getAllLabels(imagePaths_left, imagePaths_right)

for i, imagepath in enumerate(imagePaths_left):
	
	# get image name and build a name for the flipped image
	image_name = imagepath.split(os.path.sep)[-1]
	reverse_image_name = 'rev'+image_name
	
	# load and flip image horizontally
	image = cv2.imread(imagepath)
	rev_image = cv2.flip(image, 1)
	
	# get new labels
	label_left = trainLabels_left[i]
	label_right = trainLabels_right[i]
	
	if label_left.startswith('left'):
		reverse_label_right = 'right' + label_left[4:]
	elif label_left.startswith('right'):
		reverse_label_right = 'left' + label_left[5:]
	else:
		reverse_label_right = label_left
	
	
	if label_right.startswith('left'):
		reverse_label_left = 'right' + label_right[4:]
	elif label_right.startswith('right'):
		reverse_label_left = 'left'+label_right[5:]
	else:
		reverse_label_left = label_right
	
	
	
	# check if the relevant folder path exists and create if it doesn't
	dir_path = {}
	dir_path['left'] = image_folder_path['left']+reverse_label_left
	dir_path['right'] = image_folder_path['right']+reverse_label_right
	
	cv2.imwrite(dir_path['left']+'/'+reverse_image_name, rev_image)
	cv2.imwrite(dir_path['right']+'/'+reverse_image_name, rev_image)
	
	
	if i%50 == 0:
		print("[INFO] processed ",i," images successfully")
	#cv2.imshow('test',image)
	#cv2.waitKey(6)
	
