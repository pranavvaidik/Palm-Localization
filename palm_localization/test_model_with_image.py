# import packages
from config import palm_localization_config as config
import sklearn
from sklearn.preprocessing import LabelEncoder
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
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
out_image = image.copy()

# Load RGB means for traiing set
means = json.loads(open(config.DATASET_MEAN).read())

# load labele for both hands
f = open(config.OUTPUT_PATH+"/label_encoders.pkl","rb")
classes_left, classes_right = pickle.loads(f.read())
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



#print(left_output, classes_left[np.argmax(left_output)])
#print(right_output,classes_right[np.argmax(right_output)])

# show labels on image
print("press any key to exit")
cv2.putText(out_image,"left:" + classes_left[np.argmax(left_output)] + "("+ str(np.round(np.max(left_output),2)) +")", (10,25),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
cv2.putText(out_image,"right:"+classes_right[np.argmax(right_output)] + "("+ str(np.round(np.max(right_output),2)) +")", (10,55),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
cv2.imshow("Sample Output",out_image)
cv2.waitKey(0)


