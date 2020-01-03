# -*- coding: utf-8 -*-
"""
 Build a speech_dict data structure from a LibriSpeech corpus
"""

import utilities as utils
import pickle

root_dir = '../data/LibriSpeech/dev-clean'

params = {'sampling_rate': 16000,
          'with_utterances': False,
          'bunch_speaker_data':True}

speech_dict = utils.build_LibriSpeech_dict(root_dir, **params)

pickle.dump(speech_dict, open(root_dir + '/speech_dict.pkl', 'wb'))
