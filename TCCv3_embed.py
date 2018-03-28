# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:24:05 2018

@author: Vaibhav
"""
import numpy as np
import re
import pandas as pd
import nltk                                                                  
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')                                            
nltk.download('stopwords')                                                   

from sklearn.utils import shuffle
dataset = shuffle(dataset)

def mysum(i) :
    s=0
    for j in range(2,8) :
        s+=dataset.iloc[i,j]
    return s

#resampling dataset

#upsamling
for i in range(0,30000) :
    rno= np.random.randint(0,len(dataset))
    if(mysum(rno)) :
        df = pd.DataFrame(dataset.iloc[rno,:])
        pd.concat([dataset,df], axis =0)

#downsampling
indices=[]
for i in range(0,15000):
    if(mysum(i)==0) :
        indices.append(i)
dataset = dataset.drop(indices,axis=0)

def buildcorpus(dataset) :
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0,int(len(dataset))):             
        review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i,1])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

txt_corpus = buildcorpus(dataset)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D

T = Tokenizer()
T.fit_on_texts(txt_corpus)
#vocab_size = 45000
vocab_size = len(T.word_index) + 1

#integer encode the documents
encoded_docs = T.texts_to_sequences(txt_corpus)

maxlen=0
for i in range(0,len(encoded_docs)) :
    if maxlen < len(encoded_docs[i]):
        maxlen = len(encoded_docs[i])

padded_docs = pad_sequences(encoded_docs, maxlen=int(maxlen/3), padding='post')

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=int(maxlen/3)))
model.add(Convolution1D(64,3,padding = 'same'))
model.add(Convolution1D(32,3,padding = 'same'))
model.add(Convolution1D(16,3,padding = 'same'))
model.add(Convolution1D(8,3,padding = 'same'))
model.add(Flatten()) 

model.add(Dropout(rate = 0.2))

model.add(Dense(1000,kernel_initializer ='uniform',activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(300,kernel_initializer = 'uniform',activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(6, kernel_initializer = 'uniform', activation='sigmoid'))
# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

Y_t = dataset.iloc[:,2:8].values

hist = model.fit(padded_docs, Y_t, batch_size = 256, epochs = 10, validation_split = 0.2, shuffle= True)

plt.plot(hist.history['acc'],'r')
plt.plot(hist.history['val_acc'],'black')
plt.plot(hist.history['loss'],'b')
plt.plot(hist.history['val_loss'],'g')



