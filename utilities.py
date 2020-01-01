import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import librosa
import math
import random


def load_sound_files(file_paths, sr):
    raw_sounds = []
    for fp in file_paths:
        X, rates = librosa.load(fp, sr=sr)
        raw_sounds.append(X)
    return raw_sounds


def build_LibriSpeech_dict(root_dir, with_utterances=False, sampling_rate=16000, bunch_speaker_data=False):

    # Extract list of speakers
    speakers = [x[1] for x in os.walk(root_dir)][0]

    # Initialize sounds dictionary with speaker entries
    speech_dict = {}
    for speaker in speakers:
        speech_dict[speaker] = {}

    # Extract list of flac files for each speaker/chapter
    flac_files = [x[2] for x in os.walk(root_dir)]

    for entry in tqdm(flac_files):
        if len(entry) > 0:
            # Check if this entry contains flac files
            if entry[0].split('.')[-1] == 'flac':
                # Get the speaker and chapter IDs
                speaker = entry[0].split('-')[0]
                chapter = entry[0].split('-')[1]
                speech_dict[speaker][chapter] = {}

                entry.sort()
                entry_paths = []
                entry_root = root_dir + '/' + speaker + '/' + chapter
                for i in range(0,len(entry)-1):
                    entry_paths.append(entry_root + '/' + entry[i])

                if with_utterances:
                    # Store the utterances in the dictionary
                    entry_data = load_sound_files(entry_paths, sampling_rate)
                    speech_dict[speaker][chapter]['utterances'] = entry_data

                speech_dict[speaker][chapter]['flacs'] = entry_paths
                speech_dict[speaker][chapter]['trans_file'] = entry[-1]

    # Determine the size of each speaker's dataset
    for speaker in tqdm(speech_dict):        
        utterances = []
        for chapter in speech_dict[speaker]:
            if with_utterances:
                utterances = utterances + speech_dict[speaker][chapter]['utterances']        
            else:
                utterances = utterances + load_sound_files(speech_dict[speaker][chapter]['flacs'], sampling_rate)
                
        sounds = ()
        for u in utterances:
            sounds = sounds + (u,)
        
        speaker_data_all = np.concatenate(sounds, axis=0)
        n_samples = len(speaker_data_all)
        speech_dict[speaker]['n_samples'] = n_samples
        
        if bunch_speaker_data:
            outfile = root_dir + '/' + speaker + '/data_' + speaker + '.npy'
            np.save(outfile, speaker_data_all)    

    return speech_dict


def chunk_speaker_data(speech_dict, sampling_rate=16000, dt_chunk=.40, with_data=False):
    chunk_size = int(dt_chunk*sampling_rate)

    speakers = [speaker for speaker in speech_dict]

    speaker_data_chunks = {}

    for speaker in tqdm(speakers):        
        utterances = []
        for chapter in speech_dict[speaker]:
            if with_data:
                utterances = utterances + speech_dict[speaker][chapter]['utterances']        
            else:
                utterances = utterances + load_sound_files(speech_dict[speaker][chapter]['flacs'], sampling_rate)
                
        sounds = ()
        for u in utterances:
            sounds = sounds + (u,)
        
        speaker_data_all = np.concatenate(sounds, axis=0)
        n_chunks = math.floor(float(len(speaker_data_all))/float(chunk_size))
        size_to_keep = n_chunks*chunk_size
        speaker_data_chunks[speaker] = speaker_data_all[:size_to_keep].reshape((n_chunks, chunk_size))
    
    return speaker_data_chunks
    

def batch_mapping(speech_dict, chunk_size=5120, batch_size=32):
    """
    This function constructs a one-to-one mapping:
        
        batch_i --> {speaker_i, c0_i, c1_i}

    where:
        batch_i = [0, ..., n_batch-1] where n_batch = number of batches per epoch       
        speaker_i = speaker ID for batch_i
        c0_i = starting column for batch_i in the chunked data matrix for speaker_i 
        c1_i = terminating column+1 for batch_i in the chunked data matrix for speaker_i
        
    Parameters
    ----------
    speech_dict : dict
        LibriSpeech dictionary
    chunk_size : int
        size of each audio sequence chunk
        
    Returns
    -------
    batch_map : list
        [ [speaker_0, c0_0, c1_0] ,..., [speaker_{n_batch-1}, c0_{n_batch-1}, c1_{n_batch-1}] ]

    """
    
    batch_map = []
    
    for speaker in speech_dict:
        n_samples = speech_dict[speaker]['n_samples']
        n_chunks = n_samples//chunk_size
        n_batches = n_chunks//batch_size
        
        c0_i = 0
        for i in range(0, n_batches):
            c1_i = c0_i + batch_size
            batch_map.append([speaker, c0_i, c1_i])
            c0_i = c1_i
    
    return batch_map
            

def partition_speech_dict(speech_dict, chunk_size=5120, batch_size=32, min_val_frac = 0.2):
    """
        Partition speech_dict into training set and validation set
        Return new speech dicts
    """
    
    speakers = [speaker for speaker in speech_dict]
    batch_map = batch_mapping(speech_dict, chunk_size=chunk_size, batch_size=batch_size)
    n_batches = len(batch_map)
    
    val_size_target = float(n_batches)*min_val_frac
    val_size = 0
    
    speakers_val = []

    while(val_size < val_size_target):
        speaker = random.choice(speakers)
        speakers_val.append(speaker)
        speakers.remove(speaker)

        for batch_i in batch_map:
            if batch_i[0] == speaker:
                val_size = val_size + 1
        
    speakers_train = speakers
    
    speech_dict_train = {}
    for speaker in speakers_train:
        speech_dict_train[speaker] = speech_dict[speaker]
    
    speech_dict_val = {}
    for speaker in speakers_val:
        speech_dict_val[speaker] = speech_dict[speaker]
        
    val_frac = float(val_size)/float(n_batches)
    
    return speech_dict_train, speech_dict_val, val_frac
    