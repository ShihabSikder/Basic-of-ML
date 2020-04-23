# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 02:46:41 2020

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imporint dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)
# ignoring the double quote by 3

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    #removing article, preposition and conjuction
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #set works faster than list
    #now the stemming part
    #joining the list
    review = ' '.join(review)
    corpus.append(review)

#creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
#took 1500 most frequent words
#now we will run classification model 
y = dataset.iloc[:,1].values



#naive bayes classification
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here

from sklearn.naive_bayes import GaussianNB as gnb
classifier = gnb()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)




#using Random Forest
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier as rfc
classifier = rfc(n_estimators=300,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)