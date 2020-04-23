# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:19:56 2020

@author: shihab
"""
from keras.models import Sequential as SQ
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D as mx2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>=.90):
            self.model.stop_training= True
callbacks= myCallback()



#initializing CNN
classifier = SQ()

#step -1 - Convolution
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3), activation='relu'))

#step 2 pooling
classifier.add(mx2D(2,2))
#Adding Second Cov#
classifier.add(Conv2D(64,3,3,activation='relu'))
classifier.add(mx2D(2,2))
#step 3 - flattening
classifier.add(Flatten())

#step - 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compile
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Fitting CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') 

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(test_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000,
                         callbacks=[callbacks])



