import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import imutils

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat = None):
		# store the image data format
		self.dataFormat = dataFormat
		
	def preprocess(self, image):
		# apply Keras utility function to correctly rearrange the
		# dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)	

class SimplePreprocessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        # store target image dimensions and interpolation methods 
        #for resizing
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)

class AspectAwarePreprocessor:
	def __init__(self, width, height, inter = cv2.INTER_AREA):
		# store target image dimensions and interpolation methods 
        #for resizing
        self.width = width
        self.height = height
        self.inter = inter
        
	def preprocess(self, image):
		# grab the dimensions of the image and then initialize the 
		#details to use when cropping
		(h,w) = image.shape[:2]
		dW = 0
		dH = 0
		
		# if width is smaller than height, then resize along the width
		# (i.e., along the smaller dimension) and then update the deltas
		# to crop the height to the desired dimension
		if w < h:
			image = imutils.resize(image,width = self.width, inter = self.inter)
			dH = int((image.shape[0] - self.height)/2.0)
		# otherwise resize along height
		else:
			image = imutils.resize(image, height=self.height, inter=self.inter)
			dW = int((image.shape[1] - self.width)/2.0)
		
		# now regrab the dimensions and perform the crop
		(h,w) = image.shape[:2]
		image = image[dH:h-dH, dW:w-dW]
		
		# now resize the image to required spatial dimensions
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
		
class SimpleDatasetLoader:
    def __init__(self, preprocessors = None):
        # takes a list of preprocessors as inputs and stores them
        self.preprocessors = preprocessors
        
        # if preprocessors are None, initialize as empty list
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose = -1):
        # initialize the list of features and labels
        data = []
        labels = []
        
        # loop over all the image paths
        for i, imagePath in enumerate(imagePaths):
            # Load the image and it's corresponding label from the correct JSON file
            image = cv2.imread(imagePath)
            
            # get the label from the directory names
            label = imagePath.split(os.path.sep)[-2]
            
            # check if preprocessors are given
            if self.preprocessors is not None:
                # apply each of the preprocessing to the image loaded in the order
                # represented in the list
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            # append image and labels to the list of features and labels        
            data.append(image)
            labels.append(label)
            
            # show an update for every `verbose` images
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))
                
        # return a tuple of data and labels as numpy arrays
        return (np.array(data), np.array(labels))

class MultiOutputDatasetLoader:
	def __init__(self, preprocessors = None):
		# takes a list of preprocessors as inputs and stores them
		self.preprocessors = preprocessors
		
		# if preprocessors are None, initialize as empty list
		if self.preprocessors is None:
		    self.preprocessors = []
		    
	def load(self, imagePaths_left, imagePaths_right, verbose = -1):
		# initialize the list of features and labels
		data = []
		labels_left = []
		labels_right = []
		
		# loop over all the image paths
		for i, imagePath_left in enumerate(imagePaths_left):
			# get the image name
			image_name = imagePath_left.split(os.path.sep)[-1]

			# get the image path for right hand
			imagePath_right = [path for path in imagePaths_right if imagePath_left.split(os.path.sep)[-1] == image_name][0]


			# Load the image and it's corresponding label from the correct JSON file
			image = cv2.imread(imagePath_left)

			# get the label from the directory names
			label_left = imagePath_left.split(os.path.sep)[-2]
			label_right = imagePath_right.split(os.path.sep)[-2]
			#label = []

			# check if preprocessors are given
			if self.preprocessors is not None:
				# apply each of the preprocessing to the image loaded in the order
				# represented in the list
				for p in self.preprocessors:
					image = p.preprocess(image)

			# append image and labels to the list of features and labels        
			data.append(image)
			labels_left.append(label_left)
			labels_right.append(label_right)

			# show an update for every `verbose` images
			if verbose > 0 and i > 0 and (i+1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i+1,len(imagePaths_left)))
		        
		# return a tuple of data and labels as numpy arrays
		return (np.array(data), np.array(labels_left), np.array(labels_right) )
           
