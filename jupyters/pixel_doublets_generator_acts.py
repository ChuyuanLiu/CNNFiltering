import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import math
from math import pi as PI

import numpy as np
import pandas as pd

from random import shuffle

import os
from os import listdir
from os.path import isfile, join

import argparse

import time

import pickle

with open('rCuts.pickle', 'rb') as handle:
    rCuts = pickle.load(handle)
with open('zCuts.pickle', 'rb') as handle:
    zCuts = pickle.load(handle)
with open('phiCuts.pickle', 'rb') as handle:
    phiCuts = pickle.load(handle)

parser = argparse.ArgumentParser()
parser.add_argument('--graph',type=int, default=0)
parser.add_argument('--limit',type=int,default=-1)
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--path', type=str, default="/lustre/cms/store/user/adiflori/TrackML/train_1/")
parser.add_argument('--cuts', action='store_true')
parser.add_argument('--bal', action='store_true')
args = parser.parse_args()

pixelmap = {}

for i in range(0,5,1):
    pixelmap[i] = (8,i*2+2)
    
for i in range(1,8,1):
    pixelmap[i+3] = (9,i*2)
    
for i in range(7,0,-1):
    pixelmap[-i+4+7+7] = (7,i*2)
   
path = args.path
names = hits_filenames = np.array([path + f[:-9] for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("-hits.csv")))])

pixelgraph = []

for i in range(0,3):
    pixelgraph.append([i,i+1])
    
for i in range(4,10):
    pixelgraph.append([i,i+1])

for i in range(11,17):
    pixelgraph.append([i,i+1])

for i in range(0,3):
    pixelgraph.append([i,4])
    
for i in range(0,3):
    pixelgraph.append([i,11])

padsize = 15

ev = 0
hit = []
tru = []
det = []

[i,o] = pixelgraph[args.graph]
I = pixelmap[i]
O = pixelmap[o]

start = time.time()

allDoublets = []

for n in names[args.offset:args.offset+args.limit]:
    h = n + "-hits.csv"
    t = n + "-truth.csv"
    d = n + "-cells.csv"
    
    print("Loading event no. " + str(ev))
    print [i,o] 
    ev+=1
    #hit.append(pd.read_csv(h))
    #tru.append(pd.read_csv(t))
    #det.append(pd.read_csv(d))
    
    evtname = n[-14:]
    
    print evtname
    
    h = pd.read_csv(h)
    truth = pd.read_csv(t)
    detail = pd.read_csv(d)
    
    #print(truth["particle_id"].value_counts())
    
    hits  = h[((h["volume_id"]==O[0]) & (h["layer_id"]==O[1])) | ((h["volume_id"]==I[0]) & (h["layer_id"]==I[1]))]
    #out_hits = hits[(hits["volume_id"]==O[0]) & (hits["layer_id"]==O[1])]
    #[nn*chunk_size:min(len(h),chunk_size*(nn+1))]

    detail = detail[detail["hit_id"].isin(hits["hit_id"].values)]
    truth = truth[truth["hit_id"].isin(hits["hit_id"].values)]

    hits['particle_id'] = pd.Series(np.zeros(hits.values.shape[0]), index=hits.index)
    hits['particle_id'] = hits['hit_id'].map(truth.set_index('hit_id')['particle_id'])

    hits["event_id"] = pd.Series(np.full(hits.values.shape[0],float(evtname[5:])), index=hits.index)

    phis = np.arctan2(hits["y"].values, hits["x"].values)
    hits['phi'] = pd.Series(phis, index=hits.index)
    hits['px'] = hits['hit_id'].map(truth.set_index('hit_id')['tpx'])
    hits['py'] = hits['hit_id'].map(truth.set_index('hit_id')['tpy'])
    hits['pz'] = hits['hit_id'].map(truth.set_index('hit_id')['tpz'])
    hits['r']  = np.sqrt(hits["y"].values**2 + hits["x"].values**2)
    #hits['pt'] = hits['hit_id'].map(truth.set_index('hit_id')['tpt'])

    a = hits["px"].values**2
    b = hits["px"].values**2
    c = np.sqrt(a+b)
    hits["pt"] = c

    w = lambda x: round(np.average(x, weights=detail.loc[x.index, "value"]))
    M = lambda x: np.max(x)
    m = lambda x: np.min(x)

    f = {'ch0': {'avg_ch0' : w, 'max_ch0':M, 'min_ch0':m}, 'ch1': {'avg_ch1' : w, 'max_ch1':M, 'min_ch1':m}}
    new_df = detail.groupby(['hit_id'])
    b = new_df.agg(f)
    b["hit_id"] = b.index

    print "Adding clusters "

    detail = pd.merge(detail,b,on='hit_id')

    detail.columns = ["hit_id","ch0","ch1","value","avg0","min0","max0","avg1","min1","max1"]

    detail["size0"] = detail["max0"] - detail["min0"]
    detail["size1"] = detail["max1"] - detail["min1"]

    detail = detail[(detail["ch0"] - detail["avg0"]).abs()<=(padsize-1)/2]
    detail = detail[(detail["ch1"] - detail["avg1"]).abs()<=(padsize-1)/2]

    detail["pixIndex"] = 15*(detail["ch1"] - detail["avg1"] +(padsize-1)/2 ) + (detail["ch0"] - detail["avg0"] +(padsize-1)/2)

    detail = detail.drop("ch1",axis=1)
    detail = detail.drop("ch0",axis=1)

    for p in range(padsize*padsize):
        name = "Pix" + str(p)
        detail[name] = pd.Series(np.zeros(len(detail)), index=detail.index)
        detail.loc[detail["pixIndex"]==p, name] = detail.loc[detail["pixIndex"]==p, "value"]

    detail = detail.drop("pixIndex",axis=1)

    detail = detail.groupby(["hit_id","avg0","min0","max0","avg1","min1","max1","size0","size1",]).sum().reset_index()

    collabs = detail.columns
    for c in collabs[1:]:
        hits[c] = hits['hit_id'].map(detail.set_index('hit_id')[c])

    out_hits = hits[((hits["volume_id"]==O[0]) & (hits["layer_id"]==O[1]) & (hits["particle_id"]!=0))].copy() 
    in_hits  = hits[((hits["volume_id"]==I[0]) & (hits["layer_id"]==I[1]) & (hits["particle_id"]!=0))].copy()

    in_hits.columns  = ["in_" + f for f in in_hits.columns]
    out_hits.columns = ["out_" + f for f in out_hits.columns] 

    in_hits["key"]  = in_hits["in_particle_id"].values.copy()
    out_hits["key"] = out_hits["out_particle_id"].values.copy()

    in_hits_comb = hits[((hits["volume_id"]==I[0]) & (hits["layer_id"]==I[1]))].copy()
    in_hits_comb = in_hits_comb.sample(n=min(1000,len(in_hits_comb))).copy()
    out_hits_comb = hits[((hits["volume_id"]==O[0]) & (hits["layer_id"]==O[1]))].copy()
    out_hits_comb = out_hits_comb.sample(n=min(1000,len(out_hits_comb))).copy()
    in_hits_comb.columns  = ["in_" + f for f in in_hits_comb.columns]
    out_hits_comb.columns = ["out_" + f for f in out_hits_comb.columns] 
    in_hits_comb["key"]  = 1.0
    out_hits_comb["key"] = 1.0
    
    in_hits = in_hits[in_hits["key"]!=0]
    out_hits = out_hits[out_hits["key"]!=0]

    print "True doublets "

    theDoublets_true = pd.merge(in_hits,out_hits,on='key')
    theDoublets_true = theDoublets_true[np.logical_not(np.isnan(theDoublets_true["in_x"]))]

    theDoublets_true["inDet"] = pd.Series(np.full(len(theDoublets_true),i), index=theDoublets_true.index)
    theDoublets_true["outDet"] = pd.Series(np.full(len(theDoublets_true),o), index=theDoublets_true.index)

    theDoublets_true["true"] = (theDoublets_true["in_particle_id"].values-theDoublets_true["out_particle_id"].values).copy()
    theDoublets_true.loc[theDoublets_true['in_particle_id'] != theDoublets_true['out_particle_id'], ["true"]] = -1.0
    theDoublets_true.loc[theDoublets_true['in_particle_id'] == 0, ["true"]] = -2.0
    theDoublets_true.loc[theDoublets_true['out_particle_id'] == 0, ["true"]] = -3.0
    theDoublets_true.loc[(theDoublets_true['in_particle_id'] == 0) & (theDoublets_true['out_particle_id'] == 0), ["true"]] = -4.0

    theDoublets_true["deltaPhi"] = (theDoublets_true["out_phi"].values-theDoublets_true["in_phi"].values).copy()
    theDoublets_true["deltaR"] = (theDoublets_true["out_r"].values-theDoublets_true["in_r"].values).copy()
    theDoublets_true["deltaZ"] = (theDoublets_true["out_z"].values-theDoublets_true["in_z"].values).copy()
    
    theDoublets_true.loc[theDoublets_true["deltaPhi"]>PI,["deltaPhi"]]  -= 2*PI
    theDoublets_true.loc[theDoublets_true["deltaPhi"]<-PI,["deltaPhi"]] += 2*PI

    print "Comb doublets "

    theDoublets_comb = pd.merge(in_hits_comb,out_hits_comb,on='key')

    theDoublets_comb["inDet"] = pd.Series(np.full(len(theDoublets_comb),i), index=theDoublets_comb.index)
    theDoublets_comb["outDet"] = pd.Series(np.full(len(theDoublets_comb),o), index=theDoublets_comb.index)

    theDoublets_comb["true"] = (theDoublets_comb["in_particle_id"].values-theDoublets_comb["out_particle_id"].values).copy()
    theDoublets_comb.loc[theDoublets_comb['in_particle_id'] != theDoublets_comb['out_particle_id'], ["true"]] = -1.0
    theDoublets_comb.loc[theDoublets_comb['in_particle_id'] == 0, ["true"]] = -2.0
    theDoublets_comb.loc[theDoublets_comb['out_particle_id'] == 0, ["true"]] = -3.0
    theDoublets_comb.loc[(theDoublets_comb['in_particle_id'] == 0) & (theDoublets_comb['out_particle_id'] == 0), ["true"]] = -4.0

    theDoublets_comb["deltaPhi"] = (theDoublets_comb["out_phi"].values-theDoublets_comb["in_phi"].values).copy()
    theDoublets_comb["deltaR"] = (theDoublets_comb["out_r"].values-theDoublets_comb["in_r"].values).copy()
    theDoublets_comb["deltaZ"] = (theDoublets_comb["out_z"].values-theDoublets_comb["in_z"].values).copy()
    
    theDoublets_comb.loc[theDoublets_comb["deltaPhi"]>PI,["deltaPhi"]]  -= 2*PI
    theDoublets_comb.loc[theDoublets_comb["deltaPhi"]<-PI,["deltaPhi"]] += 2*PI

    print "Concat"

    theDoublets = pd.concat([theDoublets_true,theDoublets_comb])
    
    if args.cuts:
        print "Cuts"
        theDoublets = theDoublets[(theDoublets["deltaPhi"]>phiCuts[(i,o)][0]) & (theDoublets["deltaPhi"]<phiCuts[(i,o)][1])]
        theDoublets = theDoublets[(theDoublets["deltaR"]>rCuts[(i,o)][0]) & (theDoublets["deltaR"]<rCuts[(i,o)][1])]
        theDoublets = theDoublets[(theDoublets["deltaZ"]>zCuts[(i,o)][0]) & (theDoublets["deltaZ"]<zCuts[(i,o)][1])]
        
    if args.bal:
        print "bal"
        data_neg = theDoublets[theDoublets["true"] != 0.0]
        data_pos = theDoublets[theDoublets["true"] == 0.0]   
        
        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]
        
        print("Number of negatives: " + str(n_neg))
        print("Number of positive: " + str(n_pos))
        print("Ratio: " + str(float(n_neg) / float(n_pos)))
        
        n_cut = min(n_pos,n_neg)

        data_neg = data_neg.sample(n_cut)
        data_pos = data_pos.sample(n_cut)
        
        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]
        
        print("New Number of negatives: " + str(n_neg))
        print("New Number of positive: " + str(n_pos))
        print("New Ratio: " + str(n_neg / n_pos))
        
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
        theDoublets = balanced_data
      
        
    theDoublets = theDoublets.sample(frac=1.0)
    fname = evtname + "_det_" + str(i) + "_" + str(o) 
    if args.cuts:
        fname += "_cuts"
   
    if args.bal:
        fname += "_bal"
        
    fname += "_doublets.h5"
    theDoublets.to_hdf(fname,"data",append=False)
    
    allDoublets.append(theDoublets.copy())
       
fname = "all_det_" + str(i) + "_" + str(o) + "_" + str(args.offset) + "_" + str(args.limit)
if args.cuts:
    fname += "_cuts"
if args.bal:
    fname += "_bal"
    
fname += "_doublets.h5"

APPEND = False
if args.cuts:
    APPEND = True
    


allTheDoublets = pd.concat(allDoublets)
allTheDoublets.to_hdf(fname,"data",append=APPEND)

if not args.cuts:
    np.save("delta_R_" + str(i) + "_" + str(o) + ".npy",allTheDoublets["deltaR"].values)
    np.save("delta_Z_" + str(i) + "_" + str(o) + ".npy",allTheDoublets["deltaZ"].values)
    np.save("delta_Phi_" + str(i) + "_" + str(o) + ".npy",allTheDoublets["deltaPhi"].values)
    np.save("delta_PTs_" + str(i) + "_" + str(o) + ".npy",allTheDoublets["in_pt"].values)

elapsed = time.time() - start

print(str(elapsed) + " s")

