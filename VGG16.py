
import keras 
import tensorflow.compat.v1 as tf
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, ZeroPadding2D
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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 使用第几个GPU， 0是第一个
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
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
train = datagen.flow_from_directory(directory='C:/Users/Administrator/Dataset/data_self/Train',
                                            target_size=(224, 224),
                                            class_mode = 'categorical',
                                            batch_size = 8,
                                            subset = 'training')
#Assign numbers to each category
train.class_indices

#Validation dataset
test = datagen.flow_from_directory(directory='C:/Users/Administrator/Dataset/data_self/Test',
                                            target_size=(224, 224),
                                            class_mode = 'categorical',
                                            batch_size = 8,
                                            subset='validation')
    

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=4096,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=24, activation="softmax"))

from keras.optimizers import Adam
opt = Adam(lr=0.01)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
history = model.fit_generator(generator=train, validation_data= test, epochs=50,callbacks=[checkpoint,early])
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