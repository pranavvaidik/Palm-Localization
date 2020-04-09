import numpy as np
import os
import cv2

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

