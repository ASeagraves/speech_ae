# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:36:36 2019

@author: Andrew
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from generator import DataGenerator
import utilities as utils

import numpy as np
import pickle

# Speech dictionary
root_dir = '../data/LibriSpeech/dev-clean'
speech_dict_file = root_dir + '/dev-clean_dict.pkl'
speech_dict = pickle.load(open(speech_dict_file, 'rb'))

# Hyperparameters
dt_chunk = .32 # sec
sr = 16000 # Hz
chunk_size = int(dt_chunk*sr)
batch_size = 32
epochs = 40
min_val_frac = 0.2
lr=1e-3
enc_size1 = 1024
dec_size1 = 1024
latent_size = 512


speech_dict_train, speech_dict_val, val_frac = utils.partition_speech_dict(speech_dict,
                                                                                  chunk_size=chunk_size,
                                                                                  batch_size=batch_size,
                                                                                  min_val_frac=min_val_frac)

# Data generators
training_generator = DataGenerator(root_dir, speech_dict_train, 
                                   chunk_size=chunk_size, batch_size=batch_size)

validation_generator = DataGenerator(root_dir, speech_dict_val, 
                                     chunk_size=chunk_size, batch_size=batch_size)


# Autoencoder
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
              optimizer=tf.keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])

# checkpoint
checkfile="simple1-{epoch:02d}-{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(checkfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator, epochs=epochs,
                    callbacks=callbacks_list)

file = root_dir + '/84/data_84.npy'
x = np.load(file)        
n_chunks = len(x)//chunk_size
size_to_keep = n_chunks*chunk_size
X_test = x[:size_to_keep].reshape((n_chunks, chunk_size))
                            
X_pred = model.predict(x=X_test)
for i in range(40,80):
    print(str(X_test[100][i])+ ' ' + str(X_pred[100][i]))

pickle.dump(X_pred, open(root_dir + '/X_pred_84.pkl', 'wb'))
