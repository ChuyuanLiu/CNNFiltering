import pandas as pd
import numpy as np
import random
import datetime
from random import shuffle
#import seaborn as sns
import os

import tensorflow

import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *

import tracks
from tracks import *


DEBUG = os.name == 'nt'  # DEBUG on laptop

pdg = [-211., 211., 321., -321., 2212., -2212., 11., -11., 13., -13.]

#DEBUG = True

if DEBUG:
    print("DEBUG mode")

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200 if not DEBUG else 3,
                    help='number of epochs')
parser.add_argument('--path',type=str,default="data/bal_data/")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--log_dir', type=str, default="models/cnn_tracks")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--maxnorm', type=float, default=10.)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--flimit', type=int, default=100.0)
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--balance','--balance',action='store_true')
parser.add_argument('--fsamp',type=int,default=10)
parser.add_argument('--test',type=int,default=35)
parser.add_argument('--val',type=int,default=15)
parser.add_argument('--gepochs',type=float,default=1)
parser.add_argument('--loadw',type=str,default=None)
parser.add_argument('--phi',action='store_true')
parser.add_argument('--augm',type=int,default=1)
parser.add_argument('--limit',type=int,default=None)
parser.add_argument('--multiclass',action='store_true')
parser.add_argument('--kfolding',action='store_true')
parser.add_argument('--k',type=int,default=1)
args = parser.parse_args()


if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

def adam_small_doublet_model(n_channels,n_labels=2):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    infos = Input(shape=(len(tracks.featureLabs),), name='info_input')

    #drop = Dropout(args.dropou)(hit_shapes)
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
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(b_norm)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(32, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(n_labels, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


padshape = 16

remote_data = "/lustre/cms/store/user/adiflori/ConvTracks/PGun__n_5_e_10/dataset"
FILES = [remote_data + "/train/tracks_data/" + el for el in os.listdir(remote_data + "/train/tracks_data/")]
shuffle(FILES)
FILES = FILES[:args.flimit]
VAL_FILES = [remote_data +"/val/" + el for el in os.listdir(remote_data +"/val/")][:3]

train_tracks = Tracks(FILES,ptCut=[5.0,100.0])
val_tracks = Tracks(VAL_FILES,ptCut=[5.0,100.0])

train_tracks.clean_dataset()
#train_tracks.data_by_pt()
#train_tracks.pt_range()
train_tracks.data_by_pdg()

val_tracks.clean_dataset()
#val_tracks.pt_range()
val_tracks.data_by_pdg()

X_track, X_info, y = train_tracks.get_track_hits_layer_data()
X_val_track, X_val_info, y_val = val_tracks.get_track_hits_layer_data()


train_input_list = [X_track, X_info]
val_input_list = [X_val_track, X_val_info]


model = adam_small_doublet_model(train_input_list[0].shape[-1],n_labels=2)


callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience),
        ModelCheckpoint(args.log_dir + str(tnow) + "_test_last.h5", save_best_only=True,
                        save_weights_only=True),
        TensorBoard(log_dir="", histogram_freq=0,
                    write_graph=True, write_images=True)
		#roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
    ]


history = model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(val_input_list,y_val), callbacks=callbacks, verbose=True)
