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

# extract videos file paths
d = "../data/old data/video_0010.mp4"
#video_file_paths = [d+'/'+ filename for filename in os.listdir(d)]
video_file_paths = [d]

# Load RGB means for traiing set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize preprocessors
sp = SimplePreprocessor(224,224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


# initialize data processor
vp = VideoPredictor(model = model, preprocessors=[sp,mp,iap])
vp.load(video_file_paths)

