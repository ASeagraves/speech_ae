# -*- coding: utf-8 -*-
"""
 Driver for hyperparameter and architecture grid search
"""
from train import train
import os

# Base directory for parameter study
cwd = os.getcwd()
base_dir = cwd + '/results/nets'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Root data directory
root_dir = '../data/LibriSpeech/dev-clean'

# Speech dict file
speech_dict_file = root_dir + '/speech_dict.pkl'

# Hyperparameters
params_master = [[0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, False],
                 [0.05, 0.01, 0.01, 16000, 128, 100, 0.2, 1e-4, True],
                 [0.05, 0.025, 0.025, 16000, 128, 100, 0.2, 1e-4, True]]

# Size reduction factors for neural net layers
layer_params = [[1, 1, 16, 1, 1],
                [1, 1, 16, 1, 1],
                [1, 1, 16, 1, 1]]


# params_master = [[0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, False],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True],
#                 [0.05, 0.0, 0.0, 16000, 128, 100, 0.2, 1e-4, True]]

# layer_params = [[1, 4, 1],
#                 [1, 8, 1],
#                 [1, 16, 1],
#                 [1, 2, 4, 2, 1],
#                 [1, 4, 8, 4, 1],
#                 [1, 4, 16, 4, 1],
#                 [1, 1, 8, 1, 1],
#                 [1, 1, 16, 1, 1],
#                 [1, 1, 1, 8, 1, 1, 1],
#                 [1, 1, 1, 16, 1, 1, 1],
#                 [1, 1, 1, 32, 1, 1, 1]]

# Carry out training parameter studies
for i, params in enumerate(params_master):
    hyperparams = {'dt_chunk': params[0],
                   'dt_lwin': params[1],
                   'dt_rwin': params[2],
                   'sr': params[3],
                   'batch_size': params[4],
                   'epochs': params[5],
                   'min_val_frac': params[6],
                   'lr': params[7],
                   'read_partitions': params[8]}

    out_dir = base_dir + '/test' + str(i)
    train(root_dir, out_dir, speech_dict_file, hyperparams, layer_params[i])

