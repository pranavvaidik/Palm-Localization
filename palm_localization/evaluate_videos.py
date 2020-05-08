# import packages
from config import palm_localization_config as config
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
from tools.datasets import VideoPredictor
from tools.io import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import json
import numpy as np
import progressbar
import os
import pickle

# extract videos file paths
# d = "../data/old data/video_0010.mp4"
# video_file_paths = [d]

d = "../data/instructor test videos"
video_file_paths = [d+'/'+ filename for filename in os.listdir(d) if filename.endswith('.mp4')]

# Load RGB means for traiing set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize preprocessors
sp = SimplePreprocessor(224,224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# load labels for both hands
f = open(config.OUTPUT_PATH+"/label_encoders.pkl","rb")
classes_left, classes_right = pickle.loads(f.read())
f.close()

# initialize data processor
vp = VideoPredictor(model = model,  classes=(classes_left,classes_right),preprocessors=[sp,mp,iap])
print("Started evaluating videos")
vp.load(video_file_paths)

