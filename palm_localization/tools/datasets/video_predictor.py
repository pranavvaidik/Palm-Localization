import cv2

class VideoPredictor:
	def __init__(self, model, preprocessors=None):
		self.model = model
		self.preprocessors = preprocessors
		
		if self.preprocessors is None:
			self.preprocessors = []
			
	def load(self, videoPaths):
		
		# loop over each video
		for video_path in videoPaths:
			# get file path for JSON laebls for the video
			json_file_path = video_path[:-4]+'_predicted.json'
			
			json_labels = []
			
			# read video file
			video cv2.videoCapture(video_path)
			
			# get video metadata information
			frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = video.get(cv2.CAP_PROP_FPS)
			duration = frame_count/fps
			width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
			height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
			
			print("[INFO] processing file: ",video_path)
			
			# skip to next video if there is a problem opening the file
			if (video.isOpened()== False):  
				print("Error opening video file:",video_path)
				continue
			
			frame_number=0
			
			
			# Read until video is completed 
			while(video.isOpened()): 
				# videoture frame-by-frame 
				ret, frame = video.read()
				frame_time = frame_number/fps 
				frame_number += 1

				if ret == True:
					if self.preprocessors is not None:
						# apply each of the preprocessing to the image 
						# loaded in the order represented in the list
						for p in self.preprocessors:
							frame = p.preprocess(frame)
					
					# change data format later
					json_labels.append({'frame_number':frame_number, 'time':frame_time,'predictions': model.predict(frame)])
					
				else:
					with open(json_file_path, 'w') as fp:
						json.dump(labels_dict, fp)
						
					break
			
			
			
			
