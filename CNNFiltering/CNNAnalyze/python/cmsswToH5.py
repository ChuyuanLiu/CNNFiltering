import os
from os import listdir
from os.path import isfile, join
import sys, time
import argparse

from math import floor

padshape = 16

target_lab = "label"

headLab = ["run","evt","lumi","k","i","detSeqIn","detSeqOut","bSX","bSY","bSZ","bSdZ"]

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

particleLabs = ["label","tId","intersect"] + inHitFeature +  outHitFeature

featureLabs = inHitFeature + outHitFeature + ["diffADC"]

differences = ["deltaA", "deltaADC", "deltaS", "deltaR", "deltaPhi"]

dataLab = headLab + inHitLabs + outHitLabs + differences + particleLabs + ["dummyFlag"]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]


import pandas as pd
import numpy as np


def npDoubletsLoad(path,fileslimit,cols):
    print ("======================================================================")

    start = time.time()

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("txt","gz")) and "dnn_doublets" in f)])

    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")

    idName = ""

    for p in path.split("/"):
        if "dnn" in p:
            idName = p

    singlePath = path + "/singleEvts/"
    if not os.path.exists(singlePath):
        os.makedirs(singlePath)

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
            if cols:
                dfDoublets.columns = dataLab
            #print(dfDoublets.head())
            dfDoublets.to_hdf(singlePath + idName + "_" + d.replace(".txt",".h5"),'data',append=True)
            listdata.append(dfDoublets)

    alldata = pd.concat(listdata)

    dfDoublets.to_hdf(path + idName + "_" + "doublets.h5",'data',append=True)

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
