# -*- coding: utf-8 -*-
"""
Training a speech autoencoder using Keras/TensorFlow and FCNNs
"""
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from generator import DataGenerator
import autoencoder
import utilities as utils
import matplotlib.pyplot as plt
import pickle
import os


def train(root_dir, out_dir, speech_dict_file, hyperparams, layer_params):
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Unpack hyperparameters
    dt_chunk        = hyperparams['dt_chunk']  # sec
    dt_lwin         = hyperparams['dt_lwin']   # sec
    dt_rwin         = hyperparams['dt_rwin']   # sec
    sr              = hyperparams['sr']  # Hz
    batch_size      = hyperparams['batch_size']
    epochs          = hyperparams['epochs']
    min_val_frac    = hyperparams['min_val_frac']
    lr              = hyperparams['lr']
    read_partitions = hyperparams['read_partitions']

    # Derived hyperparameters
    chunk_size = int(dt_chunk * sr)
    lwin_size =  int(dt_lwin * sr)
    rwin_size =  int(dt_rwin * sr)

    # Pack size parameters for convenience
    params = {'chunk_size': chunk_size,
              'lwin_size' : lwin_size,
              'rwin_size' : rwin_size,
              'batch_size': batch_size}

    # Load speech dictionary and train/val split
    if read_partitions:
        speech_dict_train = pickle.load(open('speech_dict_train.pkl', 'rb'))
        speech_dict_val = pickle.load(open('speech_dict_val.pkl', 'rb'))
    else:
        speech_dict = pickle.load(open(speech_dict_file, 'rb'))
        speech_dict_train, speech_dict_val, val_frac = utils.partition_speech_dict(speech_dict, min_val_frac=min_val_frac, **params)
        pickle.dump(speech_dict_train, open('speech_dict_train.pkl', 'wb'))
        pickle.dump(speech_dict_val, open('speech_dict_val.pkl', 'wb'))

    # Data generators
    training_generator = DataGenerator(root_dir, speech_dict_train, **params)
    validation_generator = DataGenerator(root_dir, speech_dict_val, **params)

    # Autoencoder architecture
    layers = [chunk_size//layer_size_factor for layer_size_factor in layer_params]
    layers[0] = chunk_size + lwin_size + rwin_size # adjust input layer for left/right windowing
    model = autoencoder.build_model(layers)

    # Compile model
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])

    # Checkpoint
    checkfile = out_dir + '/fcnn-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(checkfile, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train model
    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=epochs,
                        callbacks=callbacks_list)

    # Plot training curves
    ax = plt.subplot(111)
    ax.plot(history.history['acc'], label='train', linestyle='--')
    ax.plot(history.history['val_acc'], label='val', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax.legend()
    plt.savefig(out_dir + '/accuracy.png')

    plt.clf()
    ax2 = plt.subplot(111)
    ax2.plot(history.history['loss'], label='train', linestyle='--')
    ax2.plot(history.history['val_loss'], label='val', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax2.legend()
    plt.savefig(out_dir + '/loss.png')

    # Save training history
    history_out = [history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']]
    pickle.dump(history_out, open(out_dir + '/history.pkl', 'wb'))

    # Save hyperparameters and layers
    hyperparams['chunk_size'] = chunk_size
    pickle.dump(hyperparams, open(out_dir + '/hyperparams.pkl', 'wb'))
    pickle.dump(layers, open(out_dir + '/layers.pkl', 'wb'))
