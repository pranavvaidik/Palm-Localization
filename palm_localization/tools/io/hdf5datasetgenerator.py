from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py

class HDF5DatasetGenerator:
	def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes =22):
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes
		
		# open HDF5 database for reading and determine teh total number of entries in the database
		self.db = h5py.File(dbPath,"r")
		self.numImages = self.db["labels_left"].shape[0]
		
	def generator(self, passes=np.inf):
		#initialize the epoch count
		epochs = 0
		
		# keep looking infinitely -- model will stop once it reaches a desired number of 
		# epochs
		while epochs < passes:
			# loop over hdf5 dataset
			for i in np.arange(0, self.numImages, self.batchSize):
				# extract images and labels
				images = self.db["images"][i:i+batchSize]
				labels_left = self.db["labels_left"][i:i+batchSize]
				labels_right = self.db["labels_right"][i:i+batchSize]
				
				# check if labels should be binarized
				if self.binarize:
					labels_left = to_categorical(labels_left, self.classes)
					labels_right = to_categorical(labels_right, self.classes)
				
				labels = [labels_left, labels_right]
										
				# check if preprocessors need to be applied
				if self.preprocessors is not None:
					# initialize list of processed images
					procImages = []
					
					for image in images:
						for p in self.preprocessors:
							image = p.preprocess(image)
							
						procImages.append(image)
					
					images = np.array(procImages)
				
				# if data augmentor exists, apply it
				if self.aug is not None:
					(images, labels_left)	= next(self.aug.flow(images, labels_left, batch_size=self.batchSize))
					
				yield (images, {"left":labels_left, "right":labels_right})
				
			# increment epochs
			epochs += 1
		
	def close():
		# close the database
		self.db.close()
