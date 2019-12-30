# -*- coding: utf-8 -*-
"""
 Build a dictionary for a LibriSpeech dataset
"""

import utilities as utils
import pickle

root_dir = "E:\\Documents\\deepgram\\LibriSpeech\\dev-clean"

params = {'sr': 16000,
          'with_data': True}

sounds = utils.build_LibriSpeech_dict(root_dir, sampling_rate=params['sr'], with_data=params['with_data'])

pickle.dump(sounds, open(root_dir + '\\dev-clean_data_dict.pkl', 'wb'))
