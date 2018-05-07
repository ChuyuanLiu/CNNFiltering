import os
from os import listdir
from os.path import isfile, join
import sys, time
import argparse

from math import floor
from dataset import *
import tracks

import pandas as pd
import numpy as np


def npDoubletsLoad(path,fileslimit,cols):
    print ("======================================================================")

    start = time.time()
    bal_dir = path + "/bal_data/"
    new_dir = path + "/unbal_data/"

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("txt","gz")) and "dnn_doublets" in f)])

    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")

    print("Balancing dataset in   : " + path)
    print("Saving unbalanced in   : " + new_dir)
    print("Saving balanced in     : " + bal_dir)

    if not os.path.exists(bal_dir):
        os.makedirs(bal_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    idName = ""

    for p in path.split("/"):
        if "runs" in p:
            idName = p


    print(idName)

    listdata = []
    for no,d in enumerate(datafiles):
        if os.stat(path + "/" + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + "/" + d, 'rb') as df:
            print("Reading file no." + str(no+1) + ": " + d)
            if d.lower().endswith(("txt")):
                dfDoublets = pd.read_table(df, sep="\t", header = None)
            if d.lower().endswith(("gz")):
                dfDoublets = pd.read_table(df, sep="\t", header = None,compression="gzip")

            print("--Dumping unbalanced data")
            dfDoublets.columns = dataLab
            dfDoublets.to_hdf(new_dir + idName + "_" + d.replace(".txt",".h5"),'data',append=False)

            ##balanceddata
            print("--Dumping balanced data")
            theData = Dataset([])
            theData.from_dataframe(dfDoublets)
            theData.balance_data()
            theData.save(bal_dir + idName + "_bal_" + d.replace(".txt",".h5"))

            print("--Dumping particles data")
            for p in particle_ids:
                pdg_dir = path
                if p==-1.0:
                    name = "fakes"
                else:
                    name = str(p)
                pdg_dir = pdg_dir + "/" + name + "/"

                if not os.path.exists(pdg_dir):
                    os.makedirs(pdg_dir)

                pdgData = Dataset([])
                pdgData.from_dataframe(dfDoublets)
                pdgData.separate_by_pdg(p)

                size = pdgData.data.shape[0]

                print(" - " + name + "\t\t : " + str(size) + " doublets")
                pdgData.save(pdg_dir + idName + "_" + str(p) + "_" + str(size) + "_"+ d.replace(".txt",".h5"))

            pdg_dir = path + "/others/"

            if not os.path.exists(pdg_dir):
                os.makedirs(pdg_dir)

            exclData = Dataset([])
            exclData.from_dataframe(dfDoublets)
            exclData.exclusive_by_pdg(particle_ids)
            size = exclData.data.shape[0]
            exclData.save(pdg_dir + idName + "_others_" + str(size) + "_"+ d.replace(".txt",".h5"))

    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))

def npTracksLoad(args):
    print ("======================================================================")

    start = time.time()
    #bal_dir = path + "/tracks_data/"
    new_dir = args.read + "/tracks_data/"

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("txt","gz")) and "cnn_tracks" in f)])

    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")
    print("Saving tracks in   : " + new_dir)

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    idName = ""

    for p in path.split("/"):
        if "runs" in p:
            idName = p


    print(idName)

    listdata = []

    for no,d in enumerate(datafiles[args.offset:args.offset+args.flimt]):
        if os.stat(path + "/" + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + "/" + d, 'rb') as df:
            print("Reading file no." + str(no+1) + ": " + d)
            if d.lower().endswith(("txt")):
                dfDoublets = pd.read_table(df, sep="\t", header = None)
            if d.lower().endswith(("gz")):
                dfDoublets = pd.read_table(df, sep="\t", header = None,compression="gzip")

            print("--Dumping unbalanced data")
            dfDoublets.columns = tracks.dataLab
            dfDoublets.to_hdf(new_dir + idName + "_tracks_" + d.replace(".txt",".h5"),'data',append=False)

    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))

def preprocess(path,fileslimit,cols,prep):
    print ("======================================================================")

    start = time.time()
    bal_dir = path + "/bal_data/"
    new_dir = path + "/unbal_data/"

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("txt","gz")) and "dnn_doublets" in f)])

    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")

    print("Balancing dataset in   : " + path)
    print("Saving unbalanced in   : " + new_dir)
    print("Saving balanced in     : " + bal_dir)

    if not os.path.exists(bal_dir):
        os.makedirs(bal_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    idName = ""

    for p in path.split("/"):
        if "runs" in p:
            idName = p


    print(idName)

    listdata = []
    for no,d in enumerate(datafiles):
        if os.stat(path + "/" + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + "/" + d, 'rb') as df:
            print("Reading file no." + str(no+1) + ": " + d)
            if d.lower().endswith(("txt")):
                dfDoublets = pd.read_table(df, sep="\t", header = None)
            if d.lower().endswith(("gz")):
                dfDoublets = pd.read_table(df, sep="\t", header = None,compression="gzip")

            print("--Dumping unbalanced data")
            dfDoublets.columns = dataLab
            dfDoublets.to_hdf(new_dir + idName + "_" + d.replace(".txt",".h5"),'data',append=False)

            ##balanceddata
            print("--Dumping balanced data")
            theData = Dataset([])
            theData.from_dataframe(dfDoublets)
            theData.balance_data()
            theData.save(bal_dir + idName + "_bal_" + d.replace(".txt",".h5"))

            print("--Dumping particles data")
            for p in particle_ids:
                pdg_dir = path
                if p==-1.0:
                    name = "fakes"
                else:
                    name = str(p)
                pdg_dir = pdg_dir + "/" + name + "/"

                if not os.path.exists(pdg_dir):
                    os.makedirs(pdg_dir)

                pdgData = Dataset([])
                pdgData.from_dataframe(dfDoublets)
                pdgData.separate_by_pdg(p)

                size = pdgData.data.shape[0]

                print(" - " + name + "\t\t : " + str(size) + " doublets")
                pdgData.save(pdg_dir + idName + "_" + str(p) + "_" + str(size) + "_"+ d.replace(".txt",".h5"))

            pdg_dir = path + "/others/"

            if not os.path.exists(pdg_dir):
                os.makedirs(pdg_dir)

            exclData = Dataset([])
            exclData.from_dataframe(dfDoublets)
            exclData.exclusive_by_pdg(particle_ids)
            size = exclData.data.shape[0]
            exclData.save(pdg_dir + idName + "_others_" + str(size) + "_"+ d.replace(".txt",".h5"))

    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="dataToHDF")
    parser.add_argument('--read', type=str, default="./",help='files path')
    parser.add_argument('--flimit', type=int, default=-1,help='max no. of files')
    parser.add_argument('--offset', type=int, default=0,help='offset no. of files')
    parser.add_argument('--tracks','-t',action='store_true')
    parser.add_argument('--columns','-c',action='store_true')
    parser.add_argument('--preprocess','-pre',action='store_true')

    #parser.add_argument('--debug', type=bool, default=False,help='debug printouts')
    args = parser.parse_args()

    if not args.tracks:
        npDoubletsLoad(args.read,args.flimit,args.columns)
    else:
        npTracksLoad(args)
