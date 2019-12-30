# TensorFlow / tf.keras
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint

# Helper Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper_classes import DataGenerator 
import utilities as utils
import pickle

# Parameters
lr = 1e-3
epochs = 3
batch_size = 16

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Paths
train_folder = 'stage_1_train_bss_png-224/'
labels_file = 'labels_dict.pkl'

# Datasets
labels = pickle.load( open(labels_file, 'rb') )
partition = utils.partition_data(labels, 0.20)
pickle.dump(partition, open("partition.pkl", "wb"))

# Parameters
params_train = {'batch_size':batch_size, 'dim':(224,224), 'n_classes':6,
                 'n_channels':3, 'shuffle':True, 'augment':True}

params_val = {'batch_size':batch_size, 'dim':(224,224), 'n_classes':6,
                 'n_channels':3, 'shuffle':True, 'augment':False}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params_train)
validation_generator = DataGenerator(partition['validation'], labels, **params_val)

mobilenet = MobileNetV2(
    weights='E:\\Documents\\kaggle\\input\\mobilenetV2-keras\\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
    input_shape=(224,224,3),
    include_top=False
)

# CNN Architecture
model = Sequential()
model.add(mobilenet)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(params_train['n_classes'], activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=lr),
              metrics=['accuracy', utils.weighted_loss])

# checkpoint
checkfile="mobileNetV2-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(checkfile, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator, epochs=epochs,
                    callbacks=callbacks_list)

# plot learning curves
plt.subplot(2,1,1)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2,1,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig('mobilenet-bss_ep-'+str(epochs)+'_batch-'+str(batch_size)+'_lr-'+str(lr)+'.png')
plt.show()
