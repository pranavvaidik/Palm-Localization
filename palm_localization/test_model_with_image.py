# import packages
from config import palm_localization_config as config
import sklearn
from sklearn.preprocessing import LabelEncoder
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
from tensorflow.keras.models import load_model
import json
import numpy as np
import progressbar
import os
import argparse
import cv2
import pickle

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="path to input image")
args = vars(ap.parse_args())

imagePath = args["image"]
image = cv2.imread(imagePath)

# Load RGB means for traiing set
means = json.loads(open(config.DATASET_MEAN).read())

# load label encoders
f = open(config.OUTPUT_PATH+"/label_encoders.pkl","rb")
class_left, class_right = pickle.loads(f.read())
f.close()


# initialize preprocessors
sp = SimplePreprocessor(224,224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# preprocess image
for p in [sp,mp,iap]:
	image = p.preprocess(image)

tensor = np.expand_dims(image, axis=0)

# predict labels
left_output, right_output = model.predict(tensor)



print(left_output, class_left[np.argmax(left_output)])
print(right_output,class_right[np.argmax(right_output)])


