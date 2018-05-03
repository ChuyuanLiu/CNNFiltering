
import pandas as pd
import numpy as np
import random
import seaborn as sns
import os

import tensorflow

import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *

import tracks
from tracks import *

def adam_small_doublet_model(n_channels,n_labels=2):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    infos = Input(shape=(len(tracks.featureLabs),), name='info_input')

    drop = Dropout(0.5)(hit_shapes)
    conv = Conv2D(32, (4, 4), activation='relu', padding='same', data_format="channels_last", name='conv1')(hit_shapes)
    conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
    b_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(b_norm)

    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
    b_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(b_norm)

    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv5')(pool)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='avgpool')(conv)

    flat = Flatten()(pool)
    concat = concatenate([flat, infos])

    b_norm = BatchNormalization()(concat)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(10.0), name='dense1')(b_norm)
    drop = Dropout(0.5)(dense)
    dense = Dense(32, activation='relu', kernel_constraint=max_norm(10.0), name='dense2')(drop)
    drop = Dropout(0.5)(dense)
    pred = Dense(n_labels, activation='softmax', kernel_constraint=max_norm(10.0), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


padshape = 16

remote_data = "/lustre/cms/store/user/adiflori/ConvTracks/PGun__n_5_e_10/dataset"
FILES = [remote_data + "/train/" + el for el in os.listdir(remote_data + "/train/")]
VAL_FILES = [remote_data +"/val/" + el for el in os.listdir(remote_data +"/val/")]

train_tracks = Tracks(FILES)
val_tracks = Tracks(VAL_FILES)

train_tracks.clean_dataset()
train_tracks.data_by_pt()
train_tracks.data_by_pdg()

val_tracks.clean_dataset()
val_tracks.data_by_pdg()

X_track, X_info, y = train_tracks.get_track_hits_data()
X_val_track, X_val_info, y_val = val_tracks.get_track_hits_data()


train_input_list = [X_track, X_info]
val_input_list = [X_val_track, X_val_info]



model = adam_small_doublet_model(train_input_list[0].shape[-1],n_labels=2)


callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint("test_last.h5", save_best_only=True,
                        save_weights_only=True),
        TensorBoard(log_dir="", histogram_freq=0,
                    write_graph=True, write_images=True)
		#roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
    ]


history = model.fit(train_input_list, y, batch_size=1024, epochs=100, shuffle=True,validation_data=(val_input_list,y_val), callbacks=callbacks, verbose=True)
