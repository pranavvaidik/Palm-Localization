import cv2
import numpy as np
import matplotlib
import json
import os


# A function to build batches



# A function to load a single image and converts it to a tensor
def path_to_tensor(img_path):
    


# A function to extract frame data from JSON file


# A function to join and correlate JSON labels and filenames
def json_to_labels(json_path, folder_path):
    
    with open(json_path) as f:
        labels_data = json.load(f)
        
    return labels_data


# A function for data augmentation
