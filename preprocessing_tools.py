import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat = None)
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
            
            # get the label from the correct JSON file based on file name
            #label = imagePath.split(os.path.sep)[-2]
            label = []
            
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
            
