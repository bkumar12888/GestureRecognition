#!/usr/bin/env python
# coding: utf-8

# # Gesture Recognition
# Ia this group project, you are going to build a 3D Conv model that will be able to predict the 5 gestures correctly. Please import the following libraries to get started.

# In[ ]:


import numpy as np
import os
from scipy.misc import imread, imresize
import datetime
import os
import cv2
import random
# We set the random seed so that the results don't vary drastically.

# In[ ]:


np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(30)


# In this block, you read the folder names for training and validation. You also set the `batch_size` here. Note that you set the batch size in such a way that you are able to use the GPU in full capacity. You keep increasing the batch size until the machine throws an error.

# In[ ]:


train_doc = np.random.permutation(open('/home/bkumar/upgrad/Project_data/test_train.csv').readlines())
val_doc = np.random.permutation(open('/home/bkumar/upgrad/Project_data/test_val.csv').readlines())
batch_size = 5
#experiment with the batch size


# ## Generator
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.

# In[ ]:


def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [0,1,2,4,7,10,14,18,22,27,28,29]
#create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = int(len(folder_list)/batch_size)
        # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            x=len(img_idx)
            #batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_data = np.zeros((batch_size,x,100,100,3))
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                     
                    #cropping an image if size height or width is 120* 160
                    height, width , channel = image.shape
                    if height == 120 or width == 120:
                        image=image[:120,:120,:]
                    #crop the images and resize them. Note that the images are of 2 different shape                     
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    #resizing all iamge to 100*100
                    image = cv2.resize(image,(100,100))
                    mean_image= image.mean()
                    print(mean_image)
                    batch_data[folder,idx,:,:,0] = batch_data[folder,idx,:,:,0]/mean_image
                   #normalise and feed in the image
                    #print(batch_data[folder,idx,:,:,0])
                    batch_data[folder,idx,:,:,1] = batch_data[folder,idx,:,:,1]/mean_image
                   #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = batch_data[folder,idx,:,:,2]/mean_image
                    #normalise and feed in the image
                
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do
        
        # write the code for the remaining data points which are left after full batches


# Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.
# In[ ]:
curr_dt_time = datetime.datetime.now()
train_path = '/home/bkumar/upgrad/Project_data/test_train'
val_path = '/home/bkumar/upgrad/Project_data/test_val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 1

# ## Model
# Here you make the model using different functionalities that Keras provides. Remember to use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D` for a 3D convolution model. You would want to use `TimeDistributed` while building a Conv2D + RNN model. Also remember that the last layer is the softmax. Design the network in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.

# In[ ]:
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D,Conv2D , MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers


# In[ ]:
from keras.regularizers import l2
optimiser = 'SGD'
#write your optimizer
model = Sequential()
model.add(Conv3D(64, (3, 3, 3), padding='same',
                 input_shape=(3,100,100,3)))
model.add(Activation('relu'))
model.add(Conv3D(64, (3, 3,3)))
model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Conv3D(128, (3, 3,3), padding='same'))
model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model.summary())


# Let us create the `train_generator` and the `val_generator` which will be used in `.fit_generator`.

# In[ ]:
#next(generator(train_path, train_doc, batch_size)
#next(generator(val_path, val_doc, batch_size)

train_generator = generator(train_path, train_doc, batch_size)
print("training generator done")
val_generator = generator(val_path, val_doc, batch_size)
print("validation generator done")

# In[ ]:


model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = 0.02
# write the REducelronplateau code here
callbacks_list = [checkpoint, LR]


# The `steps_per_epoch` and `validation_steps` are used by `fit_generator` to decide the number of next() calls it need to make.

# In[ ]:


if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1

print(steps_per_epoch)
print(validation_steps)

# Let us now fit the model. This will start training the model and with the help of the checkpoints, you'll be able to save the model at the end of each epoch.

# In[ ]:

model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# In[ ]:



