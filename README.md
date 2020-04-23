# Palm-Localization

## Abstract
This projects aims to loacate the relative positions of the palms of a human in a video or image with respect to his/her own body. One of the applications to this includes detection of source of pain for the human based on the palm location.


## Development
The model was developed in tensorflow 2.0  in python 3 environment. The model is trained using transfer learning, with the pre-trained network from the OpenPose model (https://github.com/CMU-Perceptual-Computing-Lab/openpose). However, since this model was trained using a Caffee, a pretrained mobilenet network by Rachit at. el. (https://github.com/rachit2403/Open-Pose-Keras) was used for transfer learning.

## Installation
You only need to clone the repository and install the requirements in order to use the model. However, large memory might be required if the video sizes are large (>5000 frames).

Before running the codes in the project, consider installing the dependencies from the requirements.txt file usinf the following command in a virtual environment
```
pip install -r requirements.txt
```

## How to Use


### To train on your own dataset
Which files to run for building the dataseet

Split videos to frames for training. Run split_to_frames.py
```
python3 split_to_frames.py
```

The datasets are built from build_palm_localization.py file. This can be done using the following command
```
python3 build_palm_localization.py
```

About config file. The configurationare in the config folder


training: use train_palmnet.py. It uses parameters stored in the config files for inputs and outputs 
```
python3 train_palmnet.py
```

Where the models, metrics, JSON files and other files are stored: Outputs are stored in the output folder. JSON files will be stored in the same directory as that of input videos

How to annotate the test set: run evaluate_videos.py for this
```
python3 evaluate_videos.py
```
### To test on sample image
How to use sample: run the file test_model_with_image.py
```
python3 test_model_with_image.py -i path/to/image/file
```

### To test on sample video

How to annotate sample video: Use sample_video_test.py and follow instructions on the terminal screen
```
python3 saample_video_test.py -v path/to/video/file
```

The script will ask you to provide your own JSON file in case you want to test it. However, it will throw an error if it is not in the right format.


### JSON label format
Describe the JSONn file format

### References

References to Openpose and Rachit's work.

References to Openpose and other references


