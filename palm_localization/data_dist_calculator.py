import os
from imutils import paths
from matplotlib import pyplot as plt

DATA_PATH_LEFT = "../data/left/"
DATA_PATH_RIGHT = "../data/right/"

labels_left = os.listdir(DATA_PATH_LEFT)
labels_right = os.listdir(DATA_PATH_RIGHT)



left_hist = [len(list(paths.list_images(DATA_PATH_LEFT+label))) for label in labels_left ]
right_hist = [len(list(paths.list_images(DATA_PATH_RIGHT+label))) for label in labels_right ]

#plt.hist(left_hist)



fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
plt.bar(labels_left,left_hist)
plt.title('Left')

fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
plt.bar(labels_right,right_hist)
plt.title('right')
plt.show()



