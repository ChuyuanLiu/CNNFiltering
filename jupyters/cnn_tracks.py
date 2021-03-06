import pandas as pd
import numpy as np
import random
import datetime
from random import shuffle
import argparse
import os

import tensorflow

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import concatenate, Dropout, BatchNormalization, AveragePooling2D
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras.regularizers import l1,l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

#import model_architectures

import tracks
from tracks import *

from model_architectures import *

IMAGE_SIZE = tracks.padshape

DEBUG = os.name == 'nt'  # DEBUG on laptop

pdg = [-211., 211., 321., -321., 2212., -2212., 11., -11., 13., -13.]

#DEBUG = True

if DEBUG:
    print("DEBUG mode")

defaultPath = "/lustre/cms/store/user/adiflori/ConvTracks/PGun__n_5_e_10/dataset"

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200 if not DEBUG else 3,
                    help='number of epochs')
parser.add_argument('--path',type=str,default=defaultPath)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--log_dir', type=str, default="models/cnn_tracks")
parser.add_argument('--name', type=str, default="cnn_tracks")
parser.add_argument('--maxnorm', type=float, default=10.)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--flimit', type=int, default=None)
parser.add_argument('--tlimit', type=int, default=50)
parser.add_argument('--gepochs',type=int,default=5)
parser.add_argument('--pt_up',type=float,default=100.0)
parser.add_argument('--pt_dw',type=float,default=1.0)
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('-s','--save',type=int,default=None)

parser.add_argument('--balance','--balance',action='store_true')
parser.add_argument('--fsamp',type=int,default=10)
parser.add_argument('--test',type=int,default=35)
parser.add_argument('--val',type=int,default=15)
parser.add_argument('--loadw',type=str,default=None)
parser.add_argument('--phi',action='store_true')
parser.add_argument('--augm',type=int,default=1)
parser.add_argument('--limit',type=int,default=None)
parser.add_argument('--multiclass',action='store_true')
parser.add_argument('--k_steps',type=int,default=1)

args = parser.parse_args()


if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.isdir(args.log_dir + "/" + str(t_now) + "/"):
    os.makedirs(args.log_dir + "/" + str(t_now) + "/")

def adam_small_doublet_model(n_channels,n_labels=2):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    infos = Input(shape=(len(tracks.featureLabs),), name='info_input')

    #drop = Dropout(args.dropout)(hit_shapes)
    conv = Conv2D(256, (4, 4), activation='relu', padding='same', data_format="channels_last", name='conv1')(hit_shapes)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
    b_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(b_norm)

    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
    b_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(b_norm)

    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv5')(pool)
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


padshape = tracks.padshape

remote_data = args.path
FILES = [remote_data + "/" + el for el in os.listdir(remote_data + "/")]
shuffle(FILES)

TEST_FILES = [remote_data + "/test/" + el for el in os.listdir(remote_data + "/test/")]
shuffle(TEST_FILES)

if (args.flimit is not None) and (args.flimit < len(FILES) ):
    FILES = FILES[:args.flimit]

if (args.tlimit < len(TEST_FILES)):
    TEST_FILES = TEST_FILES[:args.tlimit]

if args.debug:
    FILES = FILES[:3]
    TEST_FILES = TEST_FILES[:2]
    args.patience = 2
    args.n_epochs = 10

#VAL_FILES = [remote_data +"/val/" + el for el in os.listdir(remote_data +"/val/")][:3]

thePtCut = [args.pt_dw,args.pt_up]

all_tracks = Tracks(FILES,ptCut=thePtCut)
all_tracks.clean_dataset()
all_tracks.data_by_pdg()
all_tracks_data = all_tracks.data

val_frac = 1.0 / float(args.k_steps)

test_tracks = Tracks(TEST_FILES,ptCut=thePtCut)
test_tracks.clean_dataset()
test_tracks.data_by_pdg()

Test_track, Test_info, y_test = test_tracks.get_track_hits_layer_data()
test_input_list = [Test_track, Test_info]

thePtString = "_" + str(thePtCut[0]) + "_" + str(thePtCut[1])

prevname = None

if (args.save is not None):

    print("Saving data chunks...")
    print("- size   : " + str(args.save))
    print("- pt cut : " + str(thePtCut))

    ptCutDir = args.path + "/chunks/Pt" + thePtString

    if not os.path.isdir(args.path + "/chunks/"):
        os.makedirs(args.path + "/chunks/")

    if not os.path.isdir(ptCutDir):
        os.makedirs(ptCutDir)

    if args.save > all_tracks_data.shape[0]:
        chunk = 1
    else:
        chunk = int((all_tracks_data.shape[0]) / args.save)

    for i in range(chunk):
        first = args.save * i
        last = min(args.save * (i + 1),all_tracks_data.shape[0])

        if first > all_tracks_data.shape[0]:
            continue

        chunk_data = all_tracks_data[first:last]
        chunk_tracks = Tracks([])
        chunk_tracks.from_dataframe(chunk_data)
        chunk_tracks.save(ptCutDir + "/" + thePtString + "_" + str(i) + ".h5")

print("================= Training is starting with k folding")
for g in range(args.gepochs):
    for step in range(args.k_steps):

        fname = args.log_dir + "/" + str(t_now) + "/" + args.name + thePtString + "_"

        msk = np.random.rand(len(all_tracks_data)) < (1.0 - val_frac)
        train_data = all_tracks_data[msk]
        val_data   = all_tracks_data[~msk]

        train_tracks = Tracks([])
        train_tracks.from_dataframe(train_data)
        val_tracks = Tracks([])
        val_tracks.from_dataframe(val_data)

        train_tracks.clean_dataset()
        train_tracks.data_by_pdg()

        val_tracks.clean_dataset()
        val_tracks.data_by_pdg()

        print("Data loading . . . ")

        X_track, X_info, y = train_tracks.get_track_hits_layer_data()
        X_val_track, X_val_info, y_val = val_tracks.get_track_hits_layer_data()

        print(". . . done!")

        train_input_list = [X_track, X_info]
        val_input_list = [X_val_track, X_val_info]


        model = adam_small_doublet_model(train_input_list[0].shape[-1],n_labels=2)
        
        model.summary()

        if (step>0 or g>0) and (prevname is not None):
            print("Loading weights from iteration: " + str(step-1))
            model.load_weights(prevname)
        else:
            if args.loadw is not None:
                print("Loading weights from previous run: " + str(args.loadw))
                model.load_weights(args.loadw)

        print("Model loaded")

        callbacks = [
                EarlyStopping(monitor='val_acc', patience=args.patience),
                ModelCheckpoint(args.log_dir + str(t_now) + "_" + str(step) + "_" + args.name + "_test_last.h5", save_best_only=True,
                                save_weights_only=True),
                TensorBoard(log_dir=args.log_dir, histogram_freq=0,
                            write_graph=True, write_images=True)
                #roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
            ]

        print("k-Fold no . " +  str(step) + " (epoch" + str(g) + ")")
        history = model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(val_input_list,y_val), callbacks=callbacks, verbose=True)

        print("saving model " + fname)

        prevname = fname + "_fold_" + str(g) + "_" + str(step) + ".h5"

        loss, acc = model.evaluate(train_input_list, y, batch_size=args.batch_size)
        train_pred = model.predict(train_input_list)
        train_roc = roc_auc_score(y, train_pred)
        train_acc,t_train = max_binary_accuracy(y,train_pred,n=1000)

        print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss,train_roc))
        print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,train_acc,t_train))

        loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
        test_pred = model.predict(test_input_list)
        test_roc = roc_auc_score(y_test, test_pred)
        test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)

        print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
        print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))

        model.save_weights(prevname, overwrite=True)
