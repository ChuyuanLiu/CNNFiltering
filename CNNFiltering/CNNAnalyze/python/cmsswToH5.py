import os
from os import listdir
from os.path import isfile, join
import sys, time
import argparse

from math import floor
from dataset import *

padshape = 16

target_lab = "label"

headLab = ["run","evt","lumi","k","i","detSeqIn","detSeqOut","bSX","bSY","bSZ","bSdZ","PU"]

hitCoord = ["X","Y","Z","Phi","R"]

hitDet = ["DetSeq","IsBarrel","Layer","Ladder","Side","Disk","Panel","Module","IsFlipped","Ax1","Ax2"]

hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
            "AvgCharge","OverFlowX","OverFlowY","IsBig","IsBad","IsEdge"]

hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

hitCharge = ["SumADC"]

hitLabs = hitCoord + hitDet + hitClust + hitPixel + hitCharge

inHitLabs = [ "in" + str(i) for i in hitLabs]
outHitLabs = [ "out" + str(i) for i in hitLabs]

inPixels = [ "in" + str(i) for i in hitPixel]
outPixels = [ "out" + str(i) for i in hitPixel]


particleLabs = ["pId","tId","px","py","pz","pt","mT","eT","mSqr","pdgId",
                "charge","nTrackerHits","nTrackerLayers","phi","eta","rapidity",
                "vX","vY","vZ","dXY","dZ","bunchCrossing","isChargeMatched",
                "isSigSimMatched","sharedFraction","numAssocRecoTracks"]

hitFeatures = hitCoord + hitClust + hitCharge

inParticle = [ "in" + str(i) for i in particleLabs]
outParticle = [ "out" + str(i) for i in particleLabs]

inHitFeature  = [ "in" + str(i) for i in hitFeatures]
outHitFeature = [ "out" + str(i) for i in hitFeatures]

particleLabs = ["label","tId","intersect"] + inParticle +  outParticle

differences = ["deltaA", "deltaADC", "deltaS", "deltaR", "deltaPhi"]

featureLabs = inHitFeature + outHitFeature + differences

dataLab = headLab + inHitLabs + outHitLabs + differences + particleLabs + ["dummyFlag"]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

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



    listdata = []
    for no,d in enumerate(datafiles):
        if os.stat(path + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + d, 'rb') as df:
            print("Reading file no." + str(no+1) + ": " + d)
            if d.lower().endswith(("txt")):
                dfDoublets = pd.read_table(df, sep="\t", header = None)
            if d.lower().endswith(("gz")):
                dfDoublets = pd.read_table(df, sep="\t", header = None,compression="gzip")

            print("--Dumping unbalanced data")
            dfDoublets.columns = dataLab
            dfDoublets.to_hdf(new_dir + idName + "_" + d.replace(".txt",".h5"),'data',append=True)

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
                    pdg_dir = pdg_dir + "/fakes/"
                else:
                    pdg_dir = pdg_dir + str(p)

                pdgData = Dataset([])
                pdgData.from_dataframe(dfDoublets).separate_by_pdg(p)
                pdgData.to_hdf(pdg_dir + idName + "_" + d.replace(".txt",".h5"),'data',append=True)

    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))

    return alldata





if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="dataToHDF")
    parser.add_argument('--read', type=str, default="./",help='files path')
    parser.add_argument('--flimit', type=int, default=-1,help='max no. of files')
    parser.add_argument('--columns', type=bool, default=False,help='columns?')

    #parser.add_argument('--debug', type=bool, default=False,help='debug printouts')
    args = parser.parse_args()

    npDoubletsLoad(args.read,args.flimit,args.columns)
