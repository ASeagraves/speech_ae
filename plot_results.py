# -*- coding: utf-8 -*-
"""
 Plot reconstructed audio signals against the ground truth
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utilities as utils

root_dir = "../data/LibriSpeech/dev-clean"

# Trained model repository
simple1 = '../results/simple1/simple1-100_batch-128_lr-0.0001_chunk-800/simple1-82-3.349e-04-1.440e-01.hdf5'
model_file = simple1

# Hyperparameters
batch_size = 128
sr = 16000
dt = 0.05
chunk_size = int(sr*dt)

speaker = '2803'

file = root_dir + '/' + speaker + '/data_' + speaker + '.npy'
X_test = utils.chunkify_speaker_data(file, chunk_size=chunk_size)

# Load model from the disk
model = load_model(model_file)

# Prediction
X_pred = model.predict(x=X_test)

plt.plot(X_test[505:510].flatten())
plt.plot(X_pred[505:510].flatten())

plt.show()