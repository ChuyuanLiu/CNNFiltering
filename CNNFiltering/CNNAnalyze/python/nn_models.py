# flake8: noqa: E402, F401
"""
Doublet model with hit shapes and info features.
"""
#import socket
import argparse
import datetime
import json
import tempfile
import os
from dataset import Dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *
import sys
import numpy as np
import itertools
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200,help='number of epochs')
parser.add_argument('--path',type=str,default="data/bal_data/")
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--log_dir', type=str, default="models/cnn_doublet")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--maxnorm', type=float, default=10.)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--flip', type=float, default=1.0)
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


if args.debug==True:
	print("Debugging mode")

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

log_dir_tf = args.log_dir + "/" + str(t_now) + "/tf/"

if not os.path.exists(args.log_dir + "/" + str(t_now)):
    os.makedirs(args.log_dir + "/" + str(t_now))
if not os.path.exists(log_dir_tf):
    os.makedirs(log_dir_tf)


# "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"

remote_data = args.path
#debug_data = ['data/h5data/' + el for el in ['doublets.h5', 'doublets2.h5']]
debug_data = remote_data + "/debug/"

debug_files = [ debug_data + el for el in os.listdir(debug_data)]


print("Loading data...")
train_files = [remote_data + '/train/' +
               el for el in os.listdir(remote_data + 'train/')] if not args.debug else[remote_data + '/train/' +
                              el for el in os.listdir(remote_data + 'train/')][:1]

indices = np.arange(len(train_files))
np.random.shuffle(indices)

sizesamp = args.k if not args.debug else 1

kfoldindices_val   = indices[:sizesamp]
kfoldindices_test  = [el for el in indices if el not in kfoldindices_val]
kfoldindices_train = [el for el in indices if el not in kfoldindices_val and el not in kfoldindices_test]

train_files = np.take(train_files,kfoldindices_train)
test_files = np.take(train_files,kfoldindices_test)
val_files = np.take(train_files,kfoldindices_val)

val_data = Dataset(val_files)#,balance=args.balance)
test_data = Dataset(test_files)#,balance=args.balance)
train_data = Dataset(test_files)


histories = []


#dense_model
#conv_model
#separate_conv_doublet_model

for m in models:

    train_input_list = [X_hit, X_info]

    if m == "dense_model":
        X_val_hit, X_val_info, y_val = val_data.get_data()
        X_test_hit, X_test_info, y_test = test_data.get_data()
        X_train_hit, X_train_info, y_train = train_data.get_data()

        train_input_list = [X_hit, X_info]
        val_input_list = [X_val_hit, X_val_info]
        test_input_list = [X_test_hit, X_test_info]

        model = dense_model(args,train_input_list[0].shape[-1])

    if m == "conv_model":
        X_val_hit, X_val_info, y_val = val_data.get_data()
        X_test_hit, X_test_info, y_test = test_data.get_data()
        X_train_hit, X_train_info, y_train = train_data.get_data()

        train_input_list = [X_hit, X_info]
        val_input_list = [X_val_hit, X_val_info]
        test_input_list = [X_test_hit, X_test_info]

        model = conv_model(args,train_input_list[0].shape[-1])

    if m == "separate_conv_model":
        X_val_hit, X_val_info, y_val = val_data.get_data()
        X_test_hit, X_test_info, y_test = test_data.get_data()
        X_train_hit, X_train_info, y_train = train_data.get_data()

        train_input_list = [X_hit, X_info]
        val_input_list = [X_val_hit, X_val_info]
        test_input_list = [X_test_hit, X_test_info]

        model = separate_conv_doublet_model(args,train_input_list[0].shape[-1])

    if m == "layer_map_model":
        X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()
        X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()
        X_train_hit, X_train_info, y_train = train_data.get_layer_map_data()

        train_input_list = [X_hit, X_info]
        val_input_list = [X_val_hit, X_val_info]
        test_input_list = [X_test_hit, X_test_info]

        model = adam_small_doublet_model(args,train_input_list[0].shape[-1])

    fname = args.log_dir + "/" + str(t_now) + "/" + str(m)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience),
        ModelCheckpoint(fname + "_last.h5", save_best_only=True,
                        save_weights_only=True),
        TensorBoard(log_dir=log_dir_tf, histogram_freq=0,
                    write_graph=True, write_images=True)]


    history = model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(val_input_list,y_val), callbacks=callbacks, verbose=args.verbose)

    histories.append([history.history,m])

    loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
    test_pred = model.predict(test_input_list)
    test_roc = roc_auc_score(y_test, test_pred)
    test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
    print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
    print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))

    loss, acc = model.evaluate(train_input_list, y, batch_size=args.batch_size)
    train_pred = model.predict(train_input_list)
    train_roc = roc_auc_score(y, train_pred)
    train_acc,t_train = max_binary_accuracy(y,train_pred,n=1000)
    train_y = (train_pred[:,0] > t_train).astype(float)

    print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss,train_roc))
    print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,train_acc,t_train))

    print("saving model " + fname)
    model.save_weights(fname + ".h5", overwrite=True)

with open( fname + "_" + str(int(np.sum(donechunks))) + "_hist.pkl", 'wb') as file_hist:
    pickle.dump(histories, file_hist)
