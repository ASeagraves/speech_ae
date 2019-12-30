import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import librosa


def load_sound_files(file_paths, sr):
    raw_sounds = []
    for fp in file_paths:
        X, rates = librosa.load(fp, sr=sr)
        raw_sounds.append(X)
    return raw_sounds


def build_LibriSpeech_dict(root_dir, sampling_rate=16000, with_data=False):

    # Extract list of speakers
    speakers = [x[1] for x in os.walk(root_dir)][0]

    # Initialize sounds dictionary with speaker entries
    sounds = {}
    for speaker in speakers:
        sounds[speaker] = {}

    # Extract list of flac files for each speaker/chapter
    flac_files = [x[2] for x in os.walk(root_dir)]

    for entry in tqdm(flac_files):
        if len(entry) > 0:
            # Check if this entry contains flac files
            if entry[0].split('.')[-1] == 'flac':
                # Get the speaker and chapter IDs
                speaker = entry[0].split('-')[0]
                chapter = entry[0].split('-')[1]
                sounds[speaker][chapter] = {}

                entry_paths = []
                entry_root = root_dir + '\\' + speaker + '\\' + chapter
                for i in range(0,len(entry)-1):
                    entry_paths.append(entry_root + '\\' + entry[i])

                if with_data:
                    entry_data = load_sound_files(entry_paths, sampling_rate)
                    sounds[speaker][chapter]['utterances'] = entry_data

                sounds[speaker][chapter]['flacs'] = entry_paths
                sounds[speaker][chapter]['trans_file'] = entry[-1]

    return sounds


def partition_data(labels_dict, val_frac):
    """Split training IDs into training set and validation set.  Return as dict"""
    import math
    n_val = math.floor(float(len(labels_dict))*val_frac)
    ids = list(labels_dict.keys())
    val_ids = np.random.choice(ids, size=n_val, replace=False)
    train_ids = np.setdiff1d(ids, val_ids)
    return {'train':train_ids, 'validation':val_ids}
    

