import numpy as np
import pandas as pd
from tensorflow import keras as keras
import utilities as utils

class DataGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, speech_dict, chunk_size=5120, batch_size=32):
        """Initialize"""
        self.root_dir = root_dir
        self.speech_dict = speech_dict
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.batch_map = utils.batch_mapping(speech_dict, chunk_size=self.chunk_size, batch_size=self.batch_size)


    def __len__(self):
        """Number of batches per epoch"""
        return len(self.batch_map)


    def __getitem__(self, index):
        """Generate one batch of data"""
        
        # Grab batch info
        batch_i = self.batch_map[index]

        # Generate batch
        X = self.__data_generation(batch_i)

        return X, X        


    def __data_generation(self, batch_i):
        """Generates data containing batch_size samples"""
        
        # Unpack batch info
        speaker_i = batch_i[0]
        c0_i = batch_i[1]
        c1_i = batch_i[2]
        
        # Read speaker_i dataset
        file = self.root_dir + '/' + speaker_i + '/data_' + speaker_i + '.npy'
        x = np.load(file)
        
        n_chunks = len(x)//self.chunk_size
        size_to_keep = n_chunks*self.chunk_size
        X_all = x[:size_to_keep].reshape((n_chunks, self.chunk_size))
                
        # Grab batch_i data
        X = X_all[c0_i:c1_i]
        
        return  X
