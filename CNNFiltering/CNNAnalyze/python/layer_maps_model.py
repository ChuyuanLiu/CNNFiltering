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

#if socket.gethostname() == 'cmg-gpu1080':
#    print('locking only one GPU.')
#    import setGPU

def batch_generator(data_df,s):
	dsize = data_df.shape[0]
	print("Whole data size : " + str(dsize))
	print("Batch size      : " + str(s))
	slast = int(dsize / s)
	print("No. batches     : " + str(slast + 1))
 	#x_hit = data_list[0]
	#x_inf = data_list[1]
    #len(data_df)
	while True:
#		for i in range(slast+1): # 1875 * 32 = 60000 -> # of traini
            		#if i%10==0:
            		#    print "i = " + str(i)

		index= random.choice(len(s),1)
		df_i = data_df[index, :]
        	print(df_i.shape[0])
		doublets = Dataset([])
            	doublets = Dataset.from_dataframe(df_i)
            	X_hit, X_info, y = doublets.get_layer_map_data()
        	X_list = [X_hit,X_info]
           	print(X_hit.shape[0])
		yield (X_list,y)

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
parser.add_argument('--fsamp',type=int,default=10)
parser.add_argument('--gepochs',type=float,default=1)
parser.add_argument('--loadw',type=str,default=None)
parser.add_argument('--phi',type=str,default=None)
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

if args.phi is not None:
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
             el for el in os.listdir(remote_data + 'val/')] if not args.debug else debug_files
test_files = [remote_data + 'test/' +
             el for el in os.listdir(remote_data + 'test/')] if not args.debug else debug_files
# don't test yet. Test on evaluation.ipynb... # [ remote_data + el for el in  ["203_doublets.h5",  "22_doublets.h5",   "53_doublets.h5",  "64_doublets.h5",  "92_doublets.h5", "132_doublets.h5",  "159_doublets.h5",  "180_doublets.h5",  "206_doublets.h5",  "33_doublets.h5"]]
#test_files = val_files

# train_files = ['data/train_data.h5']
# val_files = ['data/val_data.h5']
# test_files = ['data/test_data.h5']

#train_data = Dataset(train_files)

val_data = Dataset(val_files)
test_data = Dataset(test_files)

nochunks = int((len(train_files) + args.fsamp - 1)/(args.fsamp))
indices = np.arange(len(train_files))
np.random.shuffle(indices)
print(indices)
donechunks = np.zeros(len(train_files))
endindices = []

#for i in range(nochunks):
i = 0

histories = []

print("loading test & val data . . .")
if not args.multiclass:
    if args.phi is None:
        X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()
        X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()
    else:
        X_val_hit, X_val_info, y_val = val_data.get_layer_map_data_withphi()
        X_test_hit, X_test_info, y_test = test_data.get_layer_map_data_withphi()
else:
    X_val_hit, X_val_info, y_val = val_data.get_layer_map_data_multiclass()
    X_test_hit, X_test_info, y_test = test_data.get_layer_map_data_multiclass()

while np.sum(donechunks) < len(train_files) * args.gepochs and (donechunks < args.gepochs).any():
    thisindices = indices[i*args.fsamp:(i+1)*args.fsamp]

    endindices = list(itertools.chain(endindices,thisindices))
    train_batch_file = np.take(train_files,thisindices)

    train_data = Dataset(train_batch_file)

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
        if args.phi is not None:
            X_hit, X_info, y = train_data.get_layer_map_data_withphi()
        else:
            X_hit, X_info, y = train_data.get_layer_map_data()
    else:
        X_hit, X_info, y = train_data.get_layer_map_data_multiclass()

    print("Training size: " + str(train_data.data.shape[0]))
    print("Val size: " + str(X_val_hit.shape[0]))
    print("Test size: " + str(X_test_hit.shape[0]))

    # [X_hit[:,:,:,:4], X_hit[:,:,:,4:], X_info]
    train_input_list = [X_hit, X_info]
    # [X_val_hit[:,:,:,:4], X_val_hit[:,:,:,4:], X_val_info]
    val_input_list = [X_val_hit, X_val_info]
    # [X_test_hit[:,:,:,:4], X_test_hit[:,:,:,4:], X_test_info]
    test_input_list = [X_test_hit, X_test_info]

    #print()

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
                    write_graph=True, write_images=True)
    ]

    #model.fit_generator(myGenerator(), samples_per_epoch = 60000, nb_epoch = 2, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    #model.fit_generator(batch_generator(train_data.data,args.bsamp),samples_per_epoch = args.bsamp , verbose=args.verbose,callbacks=callbacks,validation_data=(val_input_list, y_val),nb_epoch=args.n_epochs)

    history = model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,validation_data=(val_input_list, y_val), callbacks=callbacks, verbose=args.verbose)

    # Restore the best found model during validation
    #model.load_weights(fname + ".h5")

    loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


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
