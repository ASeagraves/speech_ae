from tensorflow import keras as keras
import utilities as utils

class DataGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, speech_dict, chunk_size=5120, lwin_size=0, 
                 rwin_size=0, batch_size=32):
        """Initialize"""
        self.root_dir = root_dir
        self.speech_dict = speech_dict
        self.data_sizes = {'chunk_size': chunk_size,
                           'lwin_size': lwin_size,
                           'rwin_size': rwin_size}
        self.batch_size = batch_size
        self.batch_map = utils.batch_mapping(speech_dict, **self.data_sizes, batch_size=self.batch_size)


    def __len__(self):
        """Number of batches per epoch"""
        return len(self.batch_map)


    def __getitem__(self, index):
        """Generate one batch of data"""
        
        # Grab batch info
        batch_i = self.batch_map[index]

        # Generate batch
        Xin, Xout = self.__data_generation(batch_i)

        return Xin, Xout        


    def __data_generation(self, batch_i):
        """Generate feature matrix containing a sequence of batch_size audio chunks"""
        
        # Unpack batch info
        [speaker_i, c0_i, c1_i] = batch_i
        
        # Filepath to speaker_i data array
        file = self.root_dir + '/' + speaker_i + '/data_' + speaker_i + '.npy'

        # Break the speaker audio signal into a chunked feature matrix        
        X_all_win = utils.chunkify_speaker_data(file, **self.data_sizes, with_windowing=True)
        X_all = utils.chunkify_speaker_data(file, **self.data_sizes, with_windowing=False)

        # Fetch batch_i data
        Xin = X_all_win[c0_i:c1_i] # Input features with overlapping window
        Xout = X_all[c0_i:c1_i] # Labels with non-overlapping window
        
        return Xin, Xout
