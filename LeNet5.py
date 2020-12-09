# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:04:10 2020

@author: Administrator
"""

import keras 
import tensorflow.compat.v1 as tf
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D 
from keras.utils import to_categorical 
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from tqdm import tqdm
import os
from PIL import Image

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 使用第几个GPU， 0是第一个

#Augment the images
datagen = ImageDataGenerator(
rotation_range=15,
rescale=1/255,
zoom_range=0.1,
shear_range=0.5,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1,
validation_split=0.2)


#Train dataset
train = datagen.flow_from_directory(directory='C:/Users/Administrator/Dataset/asl_dataset/Train',
                                            target_size=(28, 28),
                                            class_mode = 'categorical',
                                            batch_size = 32,
                                            subset = 'training')
#Assign numbers to each category
train.class_indices

#Validation dataset
test = datagen.flow_from_directory(directory='C:/Users/Administrator/Dataset/asl_dataset/Test',
                                            target_size=(28, 28),
                                            class_mode = 'categorical',
                                            batch_size = 32,
                                            subset='validation')
    
# LeNet-5
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(train, epochs=500, validation_data=test)
# making predictions 
# prediction = model.predict_classes(test)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_loss, test_accuracy = model.evaluate(test)

print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))