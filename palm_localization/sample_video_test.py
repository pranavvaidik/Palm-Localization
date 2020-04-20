# import packages
from config import palm_localization_config as config
import sklearn
from sklearn.preprocessing import LabelEncoder
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
from tools.datasets import VideoPredictor
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
ap.add_argument("-v","--video",required=True, help="path to input video")
args = vars(ap.parse_args())

video_path = args["video"]

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


# initialize data processor
vp = VideoPredictor(model = model,  classes=(classes_left,classes_right),preprocessors=[sp,mp,iap])
print("Started evaluating videos")
vp.load([video_path])


# ask if the user wants to see the video
while True:
	try:
		play_video_flag = {"yes":True, "no":False}[input("The video has finished annotation. Do you wish to see the annotated video? (yes/no)").lower()]
		break
	except KeyError:
		print("Invalid input. Please enter yes or no")


if play_video_flag:
	json_path = video_path[:-4]+'_predicted.json'
	json_labels = json.loads(open(json_path).read())
	
	video = cv2.VideoCapture(video_path)
	if (video.isOpened()== False):  
		print("Error opening video file:",video_path)

	frame_number=0

	# Read until video is completed 
	while(video.isOpened()): 
		# videoture frame-by-frame 
		ret, frame = video.read()
		#frame_time = frame_number/fps 
		#frame_number += 1

		if ret == True:
			prob_left = {}
			prob_right = {}
			for key in json_labels["left"].keys():
				prob_left[key] = json_labels["left"][key].pop(0)
			for key in json_labels["right"].keys():
				prob_right[key] = json_labels["right"][key].pop(0)
			
			left_label = max(prob_left, key = prob_left.get)
			right_label = max(prob_right, key = prob_right.get)

			# draw text on frame
			cv2.putText(frame,"left:" + left_label + "("+ str(np.round(prob_left[left_label],2)) +")", (10,25),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
			cv2.putText(frame,"right:"+ right_label + "("+ str(np.round(prob_right[right_label],2)) +")", (10,55),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
			# show frame
			cv2.imshow("labelled video",frame)
			cv2.waitKey(10)
			
		else:
			cv2.destroyAllWindows()
			break



