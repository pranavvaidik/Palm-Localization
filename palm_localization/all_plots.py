import pickle
from matplotlib import pyplot as plt
from config import palm_localization_config as config
import numpy as np

with open('output/history.pkl','rb') as fp:
	trends = pickle.load(fp)


# categorical accuracy left
plt.plot(trends['history']['val_left_out_categorical_accuracy'],label= "train accuracy")
plt.plot(trends['history']['left_out_categorical_accuracy'],label= "validation accuracy")
plt.xlabel("Epoch #")
plt.title("Left Palm Accuracy")
plt.legend()
plt.show()

# categorical accuracy right
plt.plot(trends['history']['val_right_out_categorical_accuracy'],label= "train accuracy")
plt.plot(trends['history']['right_out_categorical_accuracy'],label= "validation accuracy")
plt.xlabel("Epoch #")
plt.title("Rght Palm Accuracy")
plt.legend()
plt.show()

# left precision
plt.plot(trends['history']['val_left_out_precision'],label= "train precision")
plt.plot(trends['history']['left_out_precision'],label= "validation precision")
plt.xlabel("Epoch #")
plt.title("Left Palm Precision")
plt.legend()
plt.show()

# right precision
plt.plot(trends['history']['val_right_out_precision'],label= "train precision")
plt.plot(trends['history']['right_out_precision'],label= "validation precision")
plt.xlabel("Epoch #")
plt.title("Right Palm Precision")
plt.legend()
plt.show()


# left recall
plt.plot(trends['history']['val_left_out_recall'],label= "train recall")
plt.plot(trends['history']['left_out_recall'],label= "validation recall")
plt.xlabel("Epoch #")
plt.title("Left Palm Recall")
plt.legend()
plt.show()

# right recall
plt.plot(trends['history']['val_right_out_recall'],label= "train recall")
plt.plot(trends['history']['right_out_recall'],label= "validation recall")
plt.xlabel("Epoch #")
plt.title("Rght Palm Recall")
plt.legend()
plt.show()

# left f1 score
f1_left = 2* np.array(trends['history']['left_out_recall'])*np.array(trends['history']['left_out_precision'])/(np.array(trends['history']['left_out_recall']) + np.array(trends['history']['left_out_precision']))

val_f1_left = 2* np.array(trends['history']['val_left_out_recall'])*np.array(trends['history']['val_left_out_precision'])/(np.array(trends['history']['val_left_out_recall']) + np.array(trends['history']['val_left_out_precision']))

plt.plot(val_f1_left,label= "train F1 score")
plt.plot(f1_left,label= "validation F1 score")
plt.xlabel("Epoch #")
plt.title("Left Palm F1 score")
plt.legend()
plt.show()


# right f1 score
f1_right = 2* np.array(trends['history']['right_out_recall'])*np.array(trends['history']['right_out_precision'])/(np.array(trends['history']['right_out_recall']) + np.array(trends['history']['right_out_precision']))

val_f1_right = 2* np.array(trends['history']['val_right_out_recall'])*np.array(trends['history']['val_right_out_precision'])/(np.array(trends['history']['val_right_out_recall']) + np.array(trends['history']['val_right_out_precision']))

plt.plot(val_f1_right,label= "train F1 score")
plt.plot(f1_right,label= "validation F1 score")
plt.xlabel("Epoch #")
plt.title("Rght Palm F1 score")
plt.legend()
plt.show()



