import os
from dataset import Dataset
import argparse
import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
import random
import h5py
import keras
from keras.models import model_from_json
import json
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from dataset import *
import tensorflow as tf
layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

default_path="/eos/cms/store/group/phys_tracking/patatrack/seedingcnn/"

parser = argparse.ArgumentParser(prog="performanceplots")
parser.add_argument('--read', type=str, default=default_path,help='files path')
parser.add_argument('--chunk', type=int, default="50",help='chunk size')
parser.add_argument('--three',type=int,default=None,help='serach for train,val,test subdirectories')
parser.add_argument('--logdir',type=str,default="models/cnn_doublet/2017-10-09_11-00-43/",help='model directory')
parser.add_argument('--model',type=str,default="random_search_layer_maps",help='model name')
parser.add_argument('--debug',type=int,default=None,help='debug mode')
args = parser.parse_args()


DEBUG=False

dirs = [""]
if args.three is not None:
    dirs = ["val","test","train"]

chunksize   = args.chunk
remote_data = args.read
log_dir     = args.logdir
best_name   = args.model
offset = 0

print("loading model . . .")

with open(log_dir + '' + best_name + ".json") as f:
    model = model_from_json(json.load(f))
    if not os.path.exists(log_dir + '/' + best_name + '_final.h5'):
        model.load_weights(log_dir + '/' + best_name + '_last.h5')
    else:
        model.load_weights(log_dir + '/' + best_name + '_final.h5')

    model.summary()
    lastshape = model.layers[0].input.shape[-1]
    print(" . . . loaded ")

    for basedir in dirs:
        plots_files = [remote_data  + "/" + basedir + "/" + el for el in os.listdir(remote_data  + "/" + basedir + "/")]
        if args.debug is not None:
            plots_files = plots_files[:2]

        print("loadind files from :" + remote_data + basedir + " . . .")

        for chunk in  range(offset,int(((len(plots_files) + chunksize))/chunksize) + 1):

            if(min(len(plots_files),chunk*chunksize)==min(len(plots_files),chunksize*(chunk+1)) and chunk!=0):
                continue

            if DEBUG:
                p = plots_files[:2]
            else:
                p = plots_files[min(len(plots_files),chunk*chunksize):min(len(plots_files),chunksize*(chunk+1))]

            print("loading plot data...")
            plot_data = Dataset(p)
            print("processing plot data...")

            if lastshape == 24:
                X_hit, X_info, y = plot_data.get_layer_map_data(theta=True)
            if lastshape == 20:
                X_hit, X_info, y = plot_data.get_layer_map_data()
            if lastshape == 8:
                X_hit, X_info, y = plot_data.get_data()

            model.summary()
            y_pred = model.predict([X_hit, X_info])
            auc = roc_auc_score(y[:, 1], y_pred[:, 1])
            labels = y[:, 1]
            probtr = y_pred[:, 0]
            probfk = y_pred[:, 1]
            aucs = np.full(len(labels),auc)
            chunks = np.full(len(labels),chunk)
            print("outputting evaluated data...")
            outdata = pd.DataFrame()
            outdata = plot_data.data
            outdata.columns = dataLab

            print outdata["inX"].values.shape

            outdata.drop(inPixels,axis=1)
            outdata.drop(outPixels,axis=1)
            outdata['labels'] = pd.Series(labels, index=outdata.index)
            outdata['probtr'] = pd.Series(probtr, index=outdata.index)
            outdata['probfk'] = pd.Series(probfk, index=outdata.index)
            outdata['chunks'] = pd.Series(chunks, index=outdata.index)
            outdata['aucs']   = pd.Series(aucs, index=outdata.index)

            print("AUC, just for the sake of curiosity, is . . . " + str(aucs[0]))

            outplotdata = remote_data  + "/" + basedir  + "/" + best_name +  "/plots/"

            if not os.path.exists(outplotdata):
                os.makedirs(outplotdata)

            outdata.to_hdf(outplotdata + "/dataPlots_" + str(chunk) + "_" + str(basedir) + ".h5",'data', mode='w')
                    #outdata.head()
            #outdata[["detCounterIn","detCounterOut"] + inXYZ + outXYZ + ["probfk"]].to_csv(outplotdata + "/inferedOut.txt", header=None, index=None, sep='\t', mode='a')

            #with open(outplotdata + "/inferedOut.txt", "w") as text_file:
            #    np.savetxt(outplotdata + "/inferedOut.txt", outdata[["detCounterIn","detCounterOut"] + inXYZ + outXYZ + ["probfk"]].values, fmt='%f',delimiter='\t')
