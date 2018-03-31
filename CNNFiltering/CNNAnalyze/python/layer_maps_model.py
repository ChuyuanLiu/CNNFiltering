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

if args.phi:
	fname = fname + "_phi"

# "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"

remote_data = args.path
#debug_data = ['data/h5data/' + el for el in ['doublets.h5', 'doublets2.h5']]
debug_data = remote_data + "/debug/"

debug_files = [ debug_data + el for el in os.listdir(debug_data)]


print("Loading data...")
train_files = [remote_data + '/train/' +
               el for el in os.listdir(remote_data + 'train/')] if not args.debug else debug_files

val_files = [remote_data + 'val/' +
             el for el in os.listdir(remote_data + 'val/')]
shuffle(val_files)
val_files = val_files[:args.val] if not args.debug else debug_files

print(len(val_files))
test_files = [remote_data + 'test/' +
             el for el in os.listdir(remote_data + 'test/')]
shuffle(test_files)
test_files  = test_files[:args.test] if not args.debug else debug_files
# don't test yet. Test on evaluation.ipynb... # [ remote_data + el for el in  ["203_doublets.h5",  "22_doublets.h5",   "53_doublets.h5",  "64_doublets.h5",  "92_doublets.h5", "132_doublets.h5",  "159_doublets.h5",  "180_doublets.h5",  "206_doublets.h5",  "33_doublets.h5"]]
#test_files = val_files

# train_files = ['data/train_data.h5']
# val_files = ['data/val_data.h5']
# test_files = ['data/test_data.h5']

#train_data = Dataset(train_files)

val_data = Dataset(val_files)#,balance=args.balance)
test_data = Dataset(test_files)#,balance=args.balance)

nochunks = int((len(train_files) + args.fsamp - 1)/(args.fsamp))
indices = np.arange(len(train_files))
np.random.shuffle(indices)
# print(indices)
donechunks = np.zeros(len(train_files))
endindices = []

#for i in range(nochunks):
i = 0

histories = []

print("loading test & val data . . .")
if not args.multiclass:
    if args.phi:
        X_val_hit, X_val_info, y_val = val_data.get_layer_map_data_withphi()
        X_test_hit, X_test_info, y_test = test_data.get_layer_map_data_withphi()
    else:
        X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()
        X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()
else:
    X_val_hit, X_val_info, y_val = val_data.get_layer_map_data_multiclass()
    X_test_hit, X_test_info, y_test = test_data.get_layer_map_data_multiclass()


problematics = []

while np.sum(donechunks) < len(train_files) * args.gepochs and (donechunks < args.gepochs).any():

    numprobs = len(problematics)

    thisindices = indices[i*args.fsamp:(i+1)*args.fsamp]

    endindices = list(itertools.chain(endindices,thisindices))
    train_batch_file = np.take(train_files,thisindices)

    train_data = Dataset(train_batch_file).balance_data()

    if args.verbose:
        print("Iteration no. " + str(i) + " on " + str(nochunks))
        print("Using " + str(len(train_batch_file)) + " files with IDs : " + str(thisindices))
        for p in train_batch_file:
            print("- " + p)
        print("Tot no. of files : " + str(len(train_files)))
        print("Files done       : " + str(np.sum(donechunks)))
        print("Global epochs    : " + str(np.sum(donechunks)/len(train_files)))

    flipping = args.flip

    #train_data = train_data.filter('isFlippedIn', flipping).filter('isFlippedOut', flipping).balance_data()
    #val_data = val_data.filter('isFlippedIn', flipping).filter('isFlippedOut', flipping).balance_data()
    #test_data = test_data.filter('isFlippedIn', flipping).filter('isFlippedOut', flipping)

    #train_data = train_data.balance_data()
    #val_data = val_data.balance_data()
    #test_data = test_data

    print("loading train data . . .")

    if not args.multiclass:
        if args.phi:
            X_hit, X_info, y = train_data.get_layer_map_data_withphi(augmentation=args.augm)
        else:
            X_hit, X_info, y = train_data.get_layer_map_data(augmentation=args.augm)
    else:
        X_hit, X_info, y = train_data.get_layer_map_data_multiclass()

    print("Training size: " + str(train_data.data.shape[0]))
    print("Val size: " + str(X_val_hit.shape[0]))
    print("Test size: " + str(X_test_hit.shape[0]))

    # [X_hit[:,:,:,:4], X_hit[:,:,:,4:], X_info]
    t_list = [X_hit, X_info]



    if args.limit is not None:
		train_input_list = t_list[:args.limit]
    else:
		train_input_list = t_list

    train_input_list = train_input_list[:len(train_input_list)-numprobs]
    train_input_list = train_input_list + problematics

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

    # Restore the best found model during validation
    #model.load_weights(fname + ".h5")

    loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
    test_pred = model.predict(test_input_list)
    test_roc = roc_auc_score(y_test, test_pred)
    test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
    print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
    print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))

    train_pred = model.predict(train_input_list)
    train_acc,t_train = max_binary_accuracy(y,train_pred,n=1000)
    train_y = (train_pred > t_train).astype(float)
    print(train_y)
    prob_indeces = np.where(train_y!=y)
    print(list(prob_indeces[0]))
    problematics = train_input_list[list(prob_indeces[0])]
    print(problematics)
    print(len(problematics))
    print(len(problematics)/len(train_input_list))

    print("saving model " + fname)
    model.save_weights(fname + ".h5", overwrite=True)
    model.save_weights(fname + "_partial_" + str(int(np.sum(donechunks))) + "_" + str(i) + ".h5", overwrite=True)

    i = i + 1
    donechunks[thisindices] += 1.0

    histories.append([history.history,np.sum(donechunks)])

    if np.sum(donechunks) % len(train_files) == 0:

        model.save_weights(fname + "_" + str(int(np.sum(donechunks))) + ".h5", overwrite=True)
        np.random.shuffle(indices)
        #print(indices)
        i = 0

    with open( fname + "_" + str(int(np.sum(donechunks))) + "_hist.pkl", 'wb') as file_hist:
    	pickle.dump(histories, file_hist)

print("Global epochs    : " + str(np.sum(donechunks)/len(train_files)))
#print(endindices)
print("saving final model " + fname)
model.save_weights(fname + "_final.h5", overwrite=True)
