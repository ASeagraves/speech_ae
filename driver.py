# -*- coding: utf-8 -*-
"""
 Driver for autoencoder training hyperparameter studies
"""
from train import train
import os

# Base directory for parameter study
cwd = os.getcwd()
base_dir = cwd + '/results/nets'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Root data directory
root_dir = "../data/LibriSpeech/dev-clean"

# Hyperparameters
params_master = [[0.05, 16000, 128, 100, 0.2, 1e-4, False],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True],
                [0.05, 16000, 128, 100, 0.2, 1e-4, True]]

# Size reduction factors for neural net layers
layer_params = [[1, 4, 1],
                [1, 8, 1],
                [1, 16, 1],
                [1, 2, 4, 2, 1],
                [1, 4, 8, 4, 1],
                [1, 4, 16, 4, 1],
                [1, 1, 8, 1, 1],
                [1, 1, 16, 1, 1],
                [1, 1, 1, 8, 1, 1, 1],
                [1, 1, 1, 16, 1, 1, 1],
                [1, 1, 1, 32, 1, 1, 1]]

# Carry out training parameter studies
for i, params in enumerate(params_master):
    hyperparams = {'dt_chunk': params[0],
                   'sr': params[1],
                   'batch_size': params[2],
                   'epochs': params[3],
                   'min_val_frac': params[4],
                   'lr': params[5],
                   'read_partitions': params[6]}

    out_dir = base_dir + '/net' + str(i)
    train(root_dir, out_dir, hyperparams, layer_params[i])

