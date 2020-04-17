# import packages

# set the matplotlib backend so figures can be saved in background
import matplotlib
matplotlib.use("Agg")

# other imports
from config import palm_localization_config as config
from tools.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor,MeanPreprocessor
from tools.callbacks import TrainingMonitor
from tools.io import HDF5DatasetGenerator
from tools.nn.conv import PalmNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall
from tools.datasets import VideoPredictor
import json
import os

# construct training image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.1,
			height_shift_range = 0.1, shear_range = 0.15, horizontal_flip = False, 
			fill_mode = "nearest")

# load the RGB means for training set
means =  json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(224,224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug, preprocessors=[sp, mp, iap], classes=22)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=None, preprocessors=[sp, mp, iap], classes=22)
testGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=None, preprocessors=[sp, mp, iap], classes=22)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = PalmNet.build(width=224, classes=22)

# losses and weights for both branch outputs of the model
losses = {"output_1":"categorical_crossentropy","output_2":"categorical_crossentropy"}
lossWeights = {"output_2":1.0,"output_2":1.0}

# metrics for analysis
metrics = [TruePositives(name = 'tp'), FalsePositives(name = 'fp'), TrueNegatives(name = 'tn'), FalseNegatives(name = 'fn'), CategoricalAccuracy(name="categorical_accuracy"), Precision(name='precision'), Recall(name = 'recall')]

# compile the model
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=metrics)

# construct the set of callbacks
path = os.path.sep.join([ config.OUTPUT_PATH, "{}.png".format( os.getpid() ) ])
callbacks = [TrainingMonitor(path)]

# train the network
H = model.fit_generator(trainGen.generator(), 
					steps_per_epoch = trainGen.numImages//config.BATCH_SIZE,
					validation_data = valGen.generator(), 
					validation_steps=valGen.numImages//config.BATCH_SIZE,
					epochs = 25,
					max_queue_size = 4,
#					callbacks=callbacks,
					verbose=1)


# evaluate the model
test_results = model.evaluate_generator(testGen.generator(),steps = testGen.numImages//config.BATCH_SIZE ,verbose = 1)

print(test_results)

# save model to file			
print("[INFO] saving the model...")
model.save(config.MODEL_PATH, overwrite = True)

import pickle
with open(config.OUTPUT_PATH + "/history.pkl","wb") as fp:
	pickle.dump({'history': H.history, 'results': test_results}, fp)

# close the HDF5 datasets
trainGen.close()
valGen.close()
testGen.close()

# evaluate videos
d = "../data/old data"
video_file_paths = [d+'/'+ filename for filename in os.listdir(d)]

# initialize data processor
vp = VideoPredictor(model = model, processors=[sp,mp,iap])
vp.load(video_file_paths)

