# -*- coding: utf-8 -*-
"""
 Plot reconstructed audio spectrograms against the ground truth
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import librosa
import librosa.display
from librosa import util
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utilities as utils


root_dir = "../data/LibriSpeech/dev-clean"

# Results directories
val_dir = 'results/nets'

# Trained model repository
# 3-layer models
net0 = 'results/nets/net0'
net1 = 'results/nets/net1'
net2 = 'results/nets/net2'
#cases = [net0, net1, net2]

# 5-layer models
net4 = 'results/nets/net4'
net6 = 'results/nets/net6'
net7 = 'results/nets/net7'
#cases = [net4, net6, net7]

# 7-layer model
net9= 'results/nets/net9'  # 1-1-1-16-1-1-1
cases = [net2, net7, net9]

# 5-layer model trained on train-clean-100
#net10='C:\\Users\\Andrew\\Documents\\deepgram\\speech_ae\\results\\overlap_study1\\continue_test0'
#cases = [net7, net10]

# Hyperparameters
batch_size = 128
sr = 16000
dt = 0.05
chunk_size = int(sr*dt)

# Plot training curves
with_logspec=False
with_melspec=True
dt = 25 # plotting window
ds = int(sr*dt)
n_fft = 2048
tl = 20 # time offset from the left
sl = int(sr*tl)
speaker = '2803'
file = root_dir + '/' + speaker + '/data_' + speaker + '.npy'
X_test = utils.chunkify_speaker_data(file, chunk_size=chunk_size)
x_test = X_test.flatten()

# dict = pickle.load(open(root_dir + '/speech_dict.pkl', 'rb'))
# chapters = [chapter for chapter in dict[speaker]]
# file = dict[speaker][chapters[0]]['flacs'][0]
# x_test, _ = librosa.load(file, sr=sr)

time = np.arange(0,len(x_test))/sr
hop_length = 512
n_mels=128

if with_logspec:
    D = np.abs(librosa.stft(x_test[sl:sl+ds], n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.subplot(len(cases)+1,1,1)
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
    plt.colorbar()
elif with_melspec:
    S = librosa.feature.melspectrogram(x_test[sl:sl + ds], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.subplot(len(cases) + 1, 1, 1)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
    plt.colorbar()

for i, exp in enumerate(cases):
    [acc, val_acc, loss, val_loss] = pickle.load(open(exp + '/history.pkl', 'rb'))
    e_max = val_acc.index(max(val_acc))+1
    model_in = exp + '/fcnn-'+str(e_max).zfill(2)+'.hdf5'
    model = load_model(model_in)

    # Prediction
    X_pred = model.predict(x=X_test)
    x_pred = X_pred.flatten()
    time = np.arange(0, len(x_pred)) / sr
    hop_length = 512

    plt.subplot(len(cases) + 1, 1, i + 2)
    if with_logspec:
        Dp = np.abs(librosa.stft(x_pred[sl:sl + ds], n_fft=n_fft, hop_length=hop_length))
        DpB = librosa.amplitude_to_db(Dp, ref=np.max)
        librosa.display.specshow(DpB, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)

    elif with_melspec:
        S = librosa.feature.melspectrogram(x_pred[sl:sl + ds], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    plt.colorbar()

plt.show()
