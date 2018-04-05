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

kfoldindices_val   = thisindices[:sizesamp]
kfoldindices_test  = [el for el in thisindices if el not in kfoldindices_val]
kfoldindices_train = [el for el in thisindices if el not in kfoldindices_val and el not in kfoldindices_test]

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

    # Restore the best found model during validation
    #model.load_weights(fname + ".h5")

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
    # print(train_pred[1])
    # print(y[1])
    # print(train_y[1])
    print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss,train_roc))
    print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,train_acc,t_train))


# while np.sum(donechunks) < len(train_files) * args.gepochs and (donechunks < args.gepochs).any():
thisindices = indices[i*args.fsamp:(i+1)*args.fsamp]
train_batch_file = np.take(train_files,thisindices)


for i in range(0,len(thisindices)/args.k):

    kfoldindices_val   = thisindices[i*sizesamp:(i+1)*sizesamp]
    kfoldindices_train = [el for el in thisindices if el not in kfoldindices_val]

    print(kfoldindices_val)
    print(kfoldindices_train)

    train_batch_k = np.take(train_files,kfoldindices_train)
    val_batch_k = np.take(train_files,kfoldindices_val)

    train_data = Dataset(train_batch_k)
    val_data = Dataset(val_batch_k)#,balance=args.balance)

    X_hit, X_info, y = train_data.get_layer_map_data(theta=True)
    X_val_hit, X_val_info, y_val = val_data.get_layer_map_data(theta=True)


    X_hit = X_hit[:args.limit]
    X_info = X_info[:args.limit]
    y = y[:args.limit]

    if numprobs>0:
		print("Changing with " + str(numprobs) + " problematic doublets.")

		X_h = X_hit[:len(X_hit)-numprobs,:,:,:]
		X_i = X_info[:len(X_info)-numprobs,:,]
		yy = y[:len(y)-numprobs,:,]

		X_info = np.concatenate((X_i,problematics_info),axis=0)
		X_hit = np.concatenate((X_h,problematics_hit),axis=0)
		y = np.concatenate((yy,problematics_y),axis=0)

    train_input_list = [X_hit, X_info]

    # [X_val_hit[:,:,:,:4], X_val_hit[:,:,:,4:], X_val_info]
    val_input_list = [X_val_hit, X_val_info]
    # [X_test_hit[:,:,:,:4], X_test_hit[:,:,:,4:], X_test_info]
    test_input_list = [X_test_hit, X_test_info]

    if not args.multiclass:
        model = adam_small_doublet_model(args,train_input_list[0].shape[-1])
    else:
        model = small_doublet_model(args,train_input_list[0].shape[-1],len(pdg)+2)

    if args.verbose and i==0:
        model.summary()

    print('Training')

    if np.sum(donechunks)/len(train_files) >0.0:
        print("loading weights from iteration " + str(i-1) + " from " + fname)
        model.load_weights(fname + ".h5")
    else:
        if args.loadw is not None and os.path.isfile(args.loadw):
            print("loading weights from previous run from" + args.loadw)
            model.load_weights(args.loadw)

    with open(fname + ".json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience),
        ModelCheckpoint(fname + "_last.h5", save_best_only=True,
                        save_weights_only=True),
        TensorBoard(log_dir=log_dir_tf, histogram_freq=0,
                    write_graph=True, write_images=True),
		roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
    ]

    #model.fit_generator(myGenerator(), samples_per_epoch = 60000, nb_epoch = 2, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    #model.fit_generator(batch_generator(train_data.data,args.bsamp),samples_per_epoch = args.bsamp , verbose=args.verbose,callbacks=callbacks,validation_data=(val_input_list, y_val),nb_epoch=args.n_epochs)

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


#     train_data = Dataset(train_batch_file).balance_data()
#
#
# sizesamp = len(train_files)/args.k
# for i in range(0,args.k):
# 	thisindices = indices[i*sizesamp:(i+1)*sizesamp]
#
# 	train_batch_file = np.take(train_files,thisindices)
#
# 	train_data = Dataset(train_batch_file).balance_data()

    print("kFolding")
