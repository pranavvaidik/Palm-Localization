import cv2
import numpy as np
import os
import json

# check if the directory for left and right hands already exist
if not os.path.isdir('data/left'):
	os.mkdir('data/left')
if not os.path.isdir('data/right'):
	os.mkdir('data/right')

video_file_names = [file_name for file_name in os.listdir('data/') if file_name.endswith('.mp4')]
image_folder_path = {'left' : 'data/left/', 'right' : 'data/right/'}


for file_name in video_file_names:
	filepath = 'data/'+file_name
	json_path = filepath[:-4]+'.json'
	# check if relevant json file exists
	try:
		# load the json file and read the data
		with open(json_path) as f:
			labels_data = json.load(f)
		
		labels_left = labels_data['left_hand']
		labels_right = labels_data['right_hand']		
			
	except:
		
		print( '[INFO] json file for ', filepath, ' does not exist.' )
		continue

	# Get basic video metadata information
	video = cv2.VideoCapture(filepath)
	frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = video.get(cv2.CAP_PROP_FPS)
	duration = frame_count/fps
	width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
	height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

	frame_number = 0

	# Check if camera opened successfully 
	if (video.isOpened()== False):  
		print("Error opening video  file") 

	print("image path is:", image_folder_path, filepath, file_name)

	# Read until video is completed 
	while(video.isOpened()): 
		# videoture frame-by-frame 
		ret, frame = video.read()
		frame_number += 1
		
		if ret == True: 
			# get the left and right labels
			label_left = labels_left[frame_number-1]['location']
			label_right = labels_right[frame_number-1]['location']
			
			# Save the flip the frame horozontally and swap labels properly
			reverse_frame = cv2.flip(frame, 1)
			
			if label_left.startswith('left'):
				if not label_left.endswith('palm'):
					reverse_label_right = 'right' + label_left[4:]
			elif label_left.startswith('right'):
				reverse_label_right = 'left' + label_left[5:]
			else:
				reverse_label_right = label_left
			
			
			if label_right.startswith('left'):
				reverse_label_left = 'right' + label_right[4:]
			elif label_right.startswith('right'):
				if not label_right.endswith('palm'):
					reverse_label_left = 'left'+label_right[5:]
			else:
				reverse_label_left = label_right
			
			
			# check if the relevant folder path exists and create if it doesn't
			dir_path = {'left':image_folder_path['left']+label_left, 'right':image_folder_path['right']+label_right}
			dir_path['rev_left'] = image_folder_path['left']+reverse_label_left
			dir_path['rev_right'] = image_folder_path['right']+reverse_label_right
			
			
			for key in dir_path.keys():
				if not os.path.isdir(dir_path[key]):
					os.mkdir(dir_path[key])
				
#				if not os.path.isdir(dir_path['right']):
#					os.mkdir(dir_path['right'])
			
			# create a name for the image file and resize it
			frame_name = file_name[:-4] + str(frame_number).zfill(6) + '.jpg'
			reverse_frame_name = file_name[:-4] + 'rev_' + frame_name
			frame = cv2.resize(frame, (224,224), fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			reverse_frame = cv2.resize(reverse_frame, (224,224), fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			
			# Save the resulting frame
			cv2.imwrite(dir_path['left']+'/'+frame_name, frame)
			cv2.imwrite(dir_path['right']+'/'+frame_name, frame)
			#cv2.imwrite(dir_path['rev_left']+'/'+reverse_frame_name, reverse_frame)
			#cv2.imwrite(dir_path['rev_right']+'/'+reverse_frame_name, reverse_frame)			
				
			# Add verbose
			if frame_number%500 == 0:
				print("[INFO] preprocessed ",frame_number," frames so successfully")
		else:
		    break

	# When everything done, release  
	# the video videoture object 
	video.release()
	# Closes all the frames
	cv2.destroyAllWindows()   


    
