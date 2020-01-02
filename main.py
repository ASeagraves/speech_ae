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
import matplotlib.pyplot as plt
import numpy as np
import pickle
import autoencoder

# Speech dictionary
root_dir = '../data/LibriSpeech/dev-clean'
speech_dict_file = root_dir + '/dev-clean_dict.pkl'
speech_dict = pickle.load(open(speech_dict_file, 'rb'))

# Hyperparameters
dt_chunk = .05 # sec
sr = 16000 # Hz
chunk_size = int(dt_chunk*sr)
batch_size = 128
epochs = 100
min_val_frac = 0.2
lr=1e-4
enc_size1 = chunk_size//2
dec_size1 = chunk_size//2
latent_size = chunk_size//4
layers = [chunk_size, enc_size1, latent_size, latent_size, dec_size1, chunk_size]

params = {'chunk_size': chunk_size,
          'batch_size': batch_size}

# Training/validation split
speech_dict_train, speech_dict_val, val_frac = utils.partition_speech_dict(speech_dict, min_val_frac=min_val_frac, **params)

# Data generators
training_generator = DataGenerator(root_dir, speech_dict_train, **params)
validation_generator = DataGenerator(root_dir, speech_dict_val, **params)

# Autoencoder
model = Sequential()
autoencoder.build_model(model, layers)

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])

# checkpoint
checkfile="simple1-{epoch:02d}-{val_loss:.3e}-{val_acc:.3e}.hdf5"
checkpoint = ModelCheckpoint(checkfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator, epochs=epochs,
                    callbacks=callbacks_list)

# plot learning curves
plt.subplot(2,1,1)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig('simple1-'+str(epochs)+'_batch-'+str(batch_size)+'_lr-'+str(lr)+'_chunk-'+str(chunk_size)+'.png')
plt.show()
