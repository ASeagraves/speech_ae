# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:36:36 2019

@author: Andrew
"""
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
import helper_classes
import numpy as np
import math
from sklearn.model_selection import train_test_split

# Concatenate the utterances
utterances = ()
for u in sound_segments:
    utterances = utterances + (u,)

sounds = np.concatenate(utterances, axis=None)

print(sounds.shape)

import sys
sys.exit()

chunk_size = 5120


if 0:
    utterance = sounds[1]
    utterance_size = sounds[1].shape[0]
    N_chunks = math.floor(utterance_size/chunk_size)
    X = np.zeros((N_chunks,chunk_size))

    count = 0
    for i in range(0,N_chunks):
        for j in range(0,chunk_size):
            X[i,j] = utterance[count]
            count = count + 1
        
X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.33, random_state=42)
print(X_train)



if 0:
    for i in range(0,len(sounds)):
        N_chunks_utterance.append(math.floor(sounds[i].shape[0]/chunk_size))
    
    N_chunks = sum(N_chunks_utterance)
    
    X = np.zeros((N_chunks,chunk_size))
    
    chunk_count = 0
    for i in range(0,len(sounds)):
        utterance_i = sounds[i]
        count = 0
        for j in range(0, N_chunks_utterance[i]):
            for k in range(0, chunk_size):
                X[chunk_count,k] = utterance_i[count]
                count = count + 1
        chunk_count = chunk_count + 1

enc_size1 = 500
dec_size1 = 500
latent_size = 10

# CNN Architecture
model = Sequential()

# Encoder
model.add(Dense(chunk_size, activation='relu'))
model.add(Dense(enc_size1, activation='relu'))
model.add(Dense(latent_size, activation='relu'))

# Decoder
model.add(Dense(latent_size, activation='relu'))
model.add(Dense(dec_size1, activation='relu'))
model.add(Dense(chunk_size, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history = model.fit(x=X_train,y=X_train,batch_size=1)

X_pred = model.predict(x=X_test)
for i in range(0,10):
    print(str(X_test[5][i])+ ' ' + str(X_pred[5][i]))
