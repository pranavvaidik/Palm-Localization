# import
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

# We only need one of the paths to get all the images
trainPaths = imagePaths_left

#print("serious problem", len(imagePaths_left), len(imagePaths_right))

# get the labels
mdl = MultiOutputDatasetLoader()
trainLabels_left, trainLabels_right = mdl.getAllLabels(imagePaths_left, imagePaths_right)

le_left = LabelEncoder()
le_right = LabelEncoder()

trainLabels_left = le_left.fit_transform(trainLabels_left)
trainLabels_right = le_right.fit_transform(trainLabels_right)

f = open("output/label_encoders.pkl","wb")
f.write(pickle.dumps([le_left.classes_, le_right.classes_]))
f.close()

#print ("look here: ", len(imagePaths_left), len(imagePaths_right), len(trainLabels_left), len(trainLabels_right))

# perform sampling from the training set to build validation and test sets

(trainPaths, testPaths, trainLabels_left, testLabels_left, trainLabels_right, testLabels_right) = train_test_split(trainPaths, trainLabels_left, trainLabels_right, test_size = 0.2, random_state=42)

(trainPaths, valPaths, trainLabels_left, valLabels_left, trainLabels_right, valLabels_right) = train_test_split(trainPaths, trainLabels_left, trainLabels_right, test_size = 0.2, random_state=42)

left_weights = {x: (len(le_left.classes_ * len(trainLabels_left)))/list(trainLabels_left).count(x) for x in set(trainLabels_left)}
right_weights = {x: (len(le_right.classes_ * len(trainLabels_right)))/list(trainLabels_right).count(x) for x in set(trainLabels_right)}

# save class weights
f = open(config.OUTPUT_PATH+"/class_weights.pkl","wb")
f.write(pickle.dumps([left_weights, right_weights]))
f.close()


# construct a list pairing the training, validation and  testing image paths
# along with their corresponding labels and output HDF5 files
datasets = [("train", trainPaths, trainLabels_left, trainLabels_right, config.TRAIN_HDF5),
			("val", valPaths, valLabels_left, valLabels_right, config.VAL_HDF5),
			("test", testPaths, testLabels_left, testLabels_right, config.TEST_HDF5) ]
			
# initialize image preprocessor and list of RGB channel averages
sp = SimplePreprocessor(224,224)
(R,G,B) = ([],[],[])

for (dType, paths, labels_left, labels_right, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter( (len(paths), 224,224,3), outputPath)
	
	#initialize progress bar
	widgets = ["Building Dataset:", progressbar.Percentage()," ", progressbar.Bar()," ",progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()
	
	for (i, (path, label_left, label_right)) in enumerate(zip(paths, labels_left, labels_right)):
		# read and process the image
		image = cv2.imread(path)
		image = sp.preprocess(image)
		
		# if we are building a training dataset, then compute the mean of 
		# each channel in the image and update the lists
		if dType == "train":
			(b,g,r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)
			
		# add image and labels to HDF5 writer
		writer.add([image],[label_left],[label_right])
		pbar.update(i)
		
	# close HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages then serialize the means to a JSON file
print("[INFO] serializing means...")
D = {"R" : np.mean(R), "G" : np.mean(G), "B":np.mean(B)}
f = open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()


