# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:41:32 2018

@author: Vaibhav
"""
import numpy as np
import re
import pandas as pd
import nltk        
import matplotlib.pyplot as plt                                                           

dataset = pd.read_csv('train.csv')                                            
nltk.download('stopwords')                                                   

def buildcorpus(dataset, low, high) :
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(low,high):             
        review = re.sub('[^a-zA-Z]', ' ', dataset['comment_text'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(buildcorpus(dataset,0,len(dataset))).toarray()
y = dataset.iloc[:, 2:8].values

import gc
lst = [dataset]
del lst
gc.collect()

"""
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X = sc.fit_transform(X)
"""

#ANN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 1000, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10000))
classifier.add(Dropout(rate = 0.4))

# Adding the second hidden layer
classifier.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.4))

# Adding third layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.2))

# Output layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Fitting the ANN to the Training set
hist = classifier.fit(X, y, batch_size = 256, shuffle = True, validation_split = 0.2, epochs = 15)

plt.plot(hist.history['acc'],'g')
plt.plot(hist.history['val_acc'],'b')
plt.plot(hist.history['loss'],'g')
plt.plot(hist.history['val_loss'],'black')


#importing testset
import math
from sklearn.utils import shuffle
testset= pd.read_csv('test.csv')
testset = testset[testset.comment_text.notnull()]
testset = shuffle(testset)

#Preparing Final predictions on the given test set.

n_batches = 226
corpus=[]
preds = np.empty(226998,6, dtype = float )
for i in range(0,n_batches) :
    batch_size = math.floor(len(testset)/n_batches)
    corpus = corpus.append(buildcorpus(testset,i*batch_size,(i+1)* batch_size))
    X_test_given = cv.transform(corpus[i*batch_size:(i+1)*batch_size]).toarray()
    preds=np.append(preds,classifier.predict(X_test_given), axis=0)

corpus = corpus.append(buildcorpus(testset,i,len(testset)))
X_test_given  = cv.transform(corpus[i:len(testset)])
preds = np.append( preds, classifier.predict(X_test_given), axis = 0)



