from __future__ import print_function
import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm     
from scipy.misc import imread, imresize 
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import numpy as np
TRAIN_DIR = 'train'
TEST_DIR = 'val'
IMG_SIZE = 64
def create_label(label):
        #print(label)
	if label == '0':
		return np.array([1,0,0,0,0])
	elif label == '1':
		return np.array([0,1,0,0,0])
	elif label == '2':
		return np.array([0,0,1,0,0])
	elif label == '3':
		return np.array([0,0,0,1,0])
	elif label == '4':
		return np.array([0,0,0,0,1])
	


def create_train_data():
	training_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR,img)
                #print(path)
                for image in tqdm(os.listdir(path)):
			image_name=os.path.join(path,image)
			#print(image_name)
                        img_data=imread(image_name).astype(np.float32)
	                img_data =cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
                        #img_data=imresize(img_data,(IMG_SIZE,IMG_SIZE))                         
                        #print(img_data)
                        if "Left" in image_name:
			    var = '0'
                        if "Right" in image_name:
                            var ='1'
                        if "Stop" in image_name:
                            var = '2'
                        if "Thumbs_Down" in image_name:
                            var = '3'
                        if "Thumbs_Up" in image_name:
                            var = '4'
                        if "Down_new" in image_name:
                            var = '3'
                        if "Up_new" in image_name:
                            var = '4'
                        training_data.append([np.array(img_data),create_label(var)])
	shuffle(training_data)
	np.save('test_data.npy',training_data)
	return training_data
	
#create_train_data()
train_data = np.load('trainh_data.npy')
test_data = np.load('test_data.npy')
X_train = np.array([i[0] for i in train_data])
print(X_train.shape)
Y_train = np.array([i[1] for i in train_data])
print(Y_train[:10])
X_test = np.array([i[0] for i in test_data])
print(X_test.shape)
Y_test = np.array([i[1] for i in train_data])
print(Y_test.shape)
print(Y_test[:10])
