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
        """Generate feature matrix containing a sequence of batch_size audio chunks"""
        
        # Unpack batch info
        speaker_i = batch_i[0]
        c0_i = batch_i[1]
        c1_i = batch_i[2]
        
        # Filepath to speaker_i data array
        file = self.root_dir + '/' + speaker_i + '/data_' + speaker_i + '.npy'

        # Break the speaker audio signal into a chunked feature matrix        
        X_all = utils.chunkify_speaker_data(file, chunk_size=self.chunk_size)

        # Fetch batch_i data
        X = X_all[c0_i:c1_i]
        
        return X
