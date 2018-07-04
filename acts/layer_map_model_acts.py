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
from dataset_acts import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *
import sys
import numpy as np
import itertools
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
#if socket.gethostname() == 'cmg-gpu1080':
#    print('locking only one GPU.')
#    import setGPU


#
# # Instantiate the cross validator
# skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)
# # Loop through the indices the split() method returns
# for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
#     print "Training on fold " + str(index+1) + "/10..."
#     # Generate batches from indices
#     xtrain, xval = X[train_indices], X[val_indices]
#     ytrain, yval = y[train_indices], y[val_indices]
#     # Clear model, and create it
#     model = None
#     model = create_model()
#
#     # Debug message I guess
#     # print "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while..."
#
#     history = train_model(model, xtrain, ytrain, xval, yval)
#     accuracy_history = history.history['acc']
#     val_accuracy_history = history.history['val_acc']
#     print "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1])

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
parser.add_argument('--dump',action='store_true')
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

fname = args.log_dir + "/" + str(t_now) + "/acts_model_"


remote_data = args.path + "/" 

print("Loading data...")
main_files  = [remote_data + el for el in os.listdir(remote_data) if ".h5" in el]
train_files  = [remote_data + "/train/" + el for el in os.listdir(remote_data + "/train/") if ".h5" in el]
test_files  = [remote_data + "/test/" + el for el in os.listdir(remote_data + "/test/") if ".h5" in el]
val_files  = [remote_data + "/val/" + el for el in os.listdir(remote_data + "/val/") if ".h5" in el]
shuffle(main_files)

    
if args.limit is not None:
    main_files=main_files[:min(args.limit,len(main_files))]

if args.debug:
    main_files=main_files[:2]
    
#all_data = Dataset(main_files)
#train_data = Dataset(train_files)
#all_data.balance_data()
#theSize = float(len(all_data.data))

#print theSize


#test_data.balance_data()
val_data = Dataset(val_files)
val_data.data = val_data.data.sample(frac=1.0)
val_data.data = val_data.data[:min(len(val_data.data),2e5)]
#val_data.from_dataframe(all_data.data[int(theSize*0.2):int(theSize*0.3)])
print "val"
#val_data.balance_data()
#train_data = Dataset([])
#train_data.from_dataframe(all_data.data[int(theSize*0.3):])
print "train"
train_data = Dataset(train_files)
train_data.data = train_data.data.sample(frac=1.0)
train_data.data = train_data.data[:min(len(train_data.data),1e6)]
#train_data.balance_data()

print("Loading train data...")
X_hit, X_info, y = train_data.get_layer_map_data()
print("Loading val data...")
X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()


print("Training size: " + str(train_data.data.shape[0]))
print("Val size: " + str(X_val_hit.shape[0]))

train_input_list = [X_hit, X_info]
val_input_list = [X_val_hit, X_val_info]

callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience),
    ModelCheckpoint(fname + "_last.h5", save_best_only=True,
                    save_weights_only=True),
    TensorBoard(log_dir=log_dir_tf, histogram_freq=0,
                write_graph=True, write_images=True),
    #roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
]

model = adam_small_doublet_model(args,train_input_list[0].shape[-1])

history = model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(val_input_list,y_val), callbacks=callbacks, verbose=args.verbose)


loss, acc = model.evaluate(train_input_list, y, batch_size=args.batch_size)
train_pred = model.predict(train_input_list)
train_roc = roc_auc_score(y, train_pred)
train_acc,t_train = max_binary_accuracy(y,train_pred,n=1000)
train_y = (train_pred[:,0] > t_train).astype(float)
# print(train_pred[1])
# print(y[1])
# print(train_y[1])
print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss,train_roc))
print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,train_acc,t_train))

train_input_list[0] = 0
train_input_list[1] = 0
train_input_list[2] = 0

test_data = Dataset(test_files)
X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()
print("Test size: " + str(X_test_hit.shape[0]))
#test_data.from_dataframe(all_data.data[:int(theSize*0.2)])
print "test"
test_input_list = [X_test_hit, X_test_info]

loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
test_pred = model.predict(test_input_list)
test_roc = roc_auc_score(y_test, test_pred)
test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))


# print(train_y)
# print(y)

# [train_input_list[j-1] for j in set(prob_indeces[0])]
# print(len(problematics_info))
# print(len(problematics)/len(train_input_list))

print("saving model " + fname)
model.save_weights(fname + ".h5", overwrite=True)
model.save_weights(fname + "_partial_" + str(int(np.sum(donechunks))) + "_" + str(i) + ".h5", overwrite=True)



