# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:46:50 2018

@author: Vaibhav

"""
import numpy as np
import re
import pandas as pd
import math
import nltk                                                                  

dataset = pd.read_csv('train.csv')                                            
nltk.download('stopwords')                                                   

def buildcorpus(dataset,low,high) :
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

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0,shuffle=True)

# Training RFC

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=250,criterion='entropy',max_features='sqrt',random_state=0,oob_score=True)
classifier.fit(X_train,y_train)

oob=classifier.oob_score_
y_pred=classifier.predict(X_test)

#Accuracy measurement on cross validation set

squaresum=0
for i in range(0,len(X_test)) :
    for j in range(0,6):
        squaresum = squaresum + (y_test[i][j]-y_pred[i][j])*(y_test[i][j]-y_pred[i][j])
sqerror = squaresum/(4793*6)
accuracy = (1-sqerror)*100

#importing testset

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
