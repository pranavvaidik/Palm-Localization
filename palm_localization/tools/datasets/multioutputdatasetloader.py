import numpy as np
import os
import cv2

class MultiOutputDatasetLoader:
	def __init__(self, preprocessors = None):
		# takes a list of preprocessors as inputs and stores them
		self.preprocessors = preprocessors
		
		# if preprocessors are None, initialize as empty list
		if self.preprocessors is None:
		    self.preprocessors = []
	
	def getAllLabels(self, imagePaths_left, imagePaths_right):
		# initialize labels
		labels_left = []
		labels_right = []
		
		# loop over all image paths
		for i, imagePath_left in enumerate(imagePaths_left):
			# get the image name
			image_name = imagePath_left.split(os.path.sep)[-1]

			# get the image path for right hand
			imagePath_right = [path for path in imagePaths_right if imagePath_left.split(os.path.sep)[-1] == image_name][0]
			
			# get the label from the directory names
			label_left = imagePath_left.split(os.path.sep)[-2]
			label_right = imagePath_right.split(os.path.sep)[-2]
			
			# append image and labels to the list of features and labels  
			labels_left.append(label_left)
			labels_right.append(label_right)
			
		return ( labels_left, labels_right )
		    
	def load(self, imagePaths_left, imagePaths_right, verbose = -1):
		# initialize the list of features and labels
		data = []
		
		labels_left, labels_right = self.getAllLabels(imagePaths_left, imagePaths_right)
		#labels_left = []
		#labels_right = []
		
		# loop over all the image paths
		for i, imagePath_left in enumerate(imagePaths_left):
			# Load the image
			image = cv2.imread(imagePath_left)

			if self.preprocessors is not None:
				# apply each of the preprocessing to the image loaded in the order
				# represented in the list
				for p in self.preprocessors:
					image = p.preprocess(image)

			# append image and labels to the list of features and labels        
			data.append(image)

			# show an update for every `verbose` images
			if verbose > 0 and i > 0 and (i+1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i+1,len(imagePaths_left)))
		        
		# return a tuple of data and labels as numpy arrays
		return (np.array(data), np.array(labels_left), np.array(labels_right) )
           
