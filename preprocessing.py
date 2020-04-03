import cv2
import numpy as np
import os

video_file_names = [file_name for file_name in os.listdir('data/') if file_name.endswith('.mp4')]

for file_name in video_file_names:
    filepath = 'data/'+file_name
    try:
        os.mkdir('data/'+file_name[:-4])
    except:
        print("Folder already exists!!")
    
    image_path = filepath[:-4]+'/'
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

    print("image path is:", image_path, filepath, file_name)
    
    # Read until video is completed 
    while(video.isOpened()): 
        # videoture frame-by-frame 
        ret, frame = video.read()
        frame_number += 1
        if ret == True: 

            frame_name = str(frame_number).zfill(6) + '.jpg'
            
            frame = cv2.resize(frame, (224,224), fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

            # Save the resulting frame 
            cv2.imwrite(image_path+frame_name, frame)
            print(image_path+frame_name)

        else:
            break
    
    # When everything done, release  
    # the video videoture object 
    video.release()
    # Closes all the frames
    cv2.destroyAllWindows()
