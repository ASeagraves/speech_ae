# -*- coding: utf-8 -*-
"""
 Build a dictionary for a LibriSpeech dataset
"""

import utilities as utils
import pickle

root_dir = "../data/LibriSpeech/dev-clean"

params = {'sr': 16000,
          'with_utts': False,
          'with_bunch':True}

speech_dict = utils.build_LibriSpeech_dict(root_dir,  
                                           with_utterances = params['with_utts'],
                                           sampling_rate = params['sr'],
                                           bunch_speaker_data = params['with_bunch'])

pickle.dump(speech_dict, open(root_dir + '/dev-clean_dict.pkl', 'wb'))
