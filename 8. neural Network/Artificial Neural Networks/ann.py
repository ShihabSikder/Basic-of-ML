# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:08:37 2020

@author: Shihab
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>=.90):
            self.model.stop_training = True
callbacks= myCallback()
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encode categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:,1]= labelEncoder_X_1.fit_transform(X[:,1])

labelEncoder_X_2 = LabelEncoder()
X[:,2]= labelEncoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#ANN

#importing Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
#initializing the ANN
classifier = Sequential()
#adding first hidden layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu',input_dim=11))
#adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
#making predections and evaluating models
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))
#compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fit ANN to the training set
classifier.fit(X_train,y_train, epochs=1000 ,callbacks=[callbacks])

#predict
y_pred = classifier.predict(X_test);
y_pred = (y_pred>0.5)
#Evaluating 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
