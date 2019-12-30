import numpy as np
import cv2
import pandas as pd
from tensorflow import keras as keras
import utilities

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224,224), n_classes=5,
                 n_channels=1, shuffle=True, augment=False):
        """Initialize"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Grab list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialize feature matrix and labels
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Generate numpy array from png img file
            X[i,] = utilities.img_to_numpy('stage_1_train_bss_png-224/' + ID + '.png', self.n_channels, augment=self.augment)
            # Labels directly from label dictionary
            y[i,] = self.labels[ID]

        return  X, y

