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
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 8, aug=None, preprocessors=[sp, mp, iap], classes=22)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 8, aug=None, preprocessors=[sp, mp, iap], classes=22)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = PalmNet.build(width=224, classes=22)

# losses and weights for both branch outputs of the model
losses = {"left_out":"categorical_crossentropy","right_out":"categorical_crossentropy"}
lossWeights = {"left_out":1.0,"right_out":1.0}

model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([ config.OUTPUT_PATH, "{}.png".format( os.getpid() ) ])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(trainGen.generator(passes=10), 
					steps_per_epoch = trainGen.numImages//8,
					validation_data = valGen.generator(passes=10), 
					validation_steps=valGen.numImages//8,
					epochs = 3,
					max_queue_size = 2,
					callbacks=callbacks,
					verbose=1)

# save model to file			
print("[INFO] saving the model...")
model.save(config.MODEL_PATH, overwrite = True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
