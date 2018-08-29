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
from dataset import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *
import sys
import numpy as np
import itertools
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold

DEBUG = os.name == 'nt'  # DEBUG on laptop

pdg = [-211., 211., 321., -321., 2212., -2212., 11., -11., 13., -13.]
steps = ["detachedQuadStepHitDoublets","detachedTripletStepHitDoublets","initialStepHitDoubletsPreSplitting","lowPtQuadStepHitDoublets","mixedTripletStepHitDoublets","tripletElectronHitDoublets","allSteps"]

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
parser.add_argument('--step',type=str,default="detachedQuadStepHitDoublets")

args = parser.parse_args()

if args.name is None:
    args.name = input('model name: ')

if args.debug==True:
	print("Debugging mode")

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

log_dir_tf = args.log_dir + "/" + str(t_now) + "/tf/"

if not os.path.exists(args.log_dir + "/" + str(t_now)):
    os.makedirs(args.log_dir + "/" + str(t_now))
if not os.path.exists(log_dir_tf):
    os.makedirs(log_dir_tf)

fname = args.log_dir + "/" + str(t_now) + "/" + args.name

remote_data = args.path + "/" + args.step + "/"
debug_data = remote_data + "/debug/"

print("Loading data...")
main_files  = [remote_data + el for el in os.listdir(remote_data) if ".h5" in el]
debug_files = main_files[:3]
shuffle(main_files)

train_files = main_files[:int(len(main_files)*0.8)] if not args.debug else debug_files #[remote_data + '/train/' +
               #el for el in os.listdir(remote_data + 'train/')] if not args.debug else debug_files

val_files = main_files[int(len(main_files)*0.8):] if not args.debug else debug_files

test_files = [remote_data + '/test/' +
             el for el in os.listdir(remote_data + 'test/')] if not args.debug else debug_files

shuffle(test_files)

val_data = Dataset(val_files)#,balance=args.balance)
test_data = Dataset(test_files)
train_data = Dataset(train_files)

X_val_hit, y_val = val_data.first_layer_map_data()
X_test_hit, y_test = test_data.first_layer_map_data()
X_hit, y = train_data.first_layer_map_data()

model = pixel_only_model(args,X_hit.shape[-1])

with open(fname + ".json", "w") as outfile:
    json.dump(model.to_json(), outfile)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience),
    ModelCheckpoint(fname + "pixelonly_last.h5", save_best_only=True,
                    save_weights_only=True),
    TensorBoard(log_dir=log_dir_tf, histogram_freq=0,
                write_graph=True, write_images=True)
    #roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
]

#model.fit_generator(myGenerator(), samples_per_epoch = 60000, nb_epoch = 2, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
#model.fit_generator(batch_generator(train_data.data,args.bsamp),samples_per_epoch = args.bsamp , verbose=args.verbose,callbacks=callbacks,validation_data=(val_input_list, y_val),nb_epoch=args.n_epochs)

history = model.fit(X_hit, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(X_val_hit,y_val), callbacks=callbacks, verbose=args.verbose)


loss, acc = model.evaluate(X_test_hit, y_test, batch_size=args.batch_size)
test_pred = model.predict(X_test_hit)
test_roc = roc_auc_score(y_test, test_pred)
test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))

loss, acc = model.evaluate(X_hit, y, batch_size=args.batch_size)
train_pred = model.predict(X_hit)
train_roc = roc_auc_score(y, train_pred)
train_acc,t_train = max_binary_accuracy(y,train_pred,n=1000)
train_y = (train_pred[:,0] > t_train).astype(float)

print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss,train_roc))
print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,train_acc,t_train))

print("saving model " + fname)
model.save_weights(fname + "pixelonly_weight.h5", overwrite=True)

with open( fname + "pixelonly_hist.pkl", 'wb') as file_hist:
    pickle.dump(history.history, file_hist)
