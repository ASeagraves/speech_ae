# -*- coding: utf-8 -*-
"""
 Build a dictionary for a LibriSpeech dataset
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

root_dir = "../data/LibriSpeech/dev-clean"

batch_size = 32
chunk_size = 5120

x = np.load(root_dir + '/84/data_84.npy')
n_chunks = len(x)//chunk_size
size_to_keep = n_chunks*chunk_size
X_test = x[:size_to_keep].reshape((n_chunks, chunk_size))

X_pred = pickle.load(open(root_dir + '/X_pred_84.pkl', 'rb'))

case = 805

plt.plot(x)
#plt.plot(X_test[case])
#plt.plot(X_pred[case])
plt.plot(X_pred.flatten())

plt.show()