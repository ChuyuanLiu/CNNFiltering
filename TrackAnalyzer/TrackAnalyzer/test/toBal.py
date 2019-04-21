import pandas as pd
import os

import sys

import argparse

from time import time
parser = argparse.ArgumentParser()
parser.add_argument('--path',  type=str, default="./")
parser.add_argument('--split', type=str, default=None)
parser.add_argument('--partial',action="store_true")
#parser.add_argument('--dir',   type=str, default=None)
#parser.add_argument('--tree',  type=str, default="2mu2kSkimmedTree")
args = parser.parse_args()


data_files = [args.path + f for f in os.listdir(args.path) if f.endswith(".h5") and "bal" not in f]

if args.split is not None:
    data_files = [ f for f in data_files if f.endswith(args.split + ".h5")]

print("Balancing")

main_pdgs = [11.,13.,211.,321.,2212.]

for ff in data_files:

    bal_name = ff[:-3] + "_bal.h5"
    full_name = ff[:-3] + "_full_bal.h5"

    if not os.path.isfile(full_name):

            print("Loading File : " + f.split("/")[-1])

            t = time()
            tmp = 0
            tmp = pd.read_hdf(ff)

            #uproot.open(cnn_file)["TrackProducer"]["CnnTracks"].pandas.df()
            secondBest = (tmp["pdg"].value_counts().values)[1]
            lastBest = (tmp["pdg"].value_counts().values)[-1] - 1
            selection = (tmp["pdg"]>-9000.0)

            fully_balanced = tmp[~selection].sample(n=lastBest)
            for p in main_pdgs:
                fully_balanced = pd.concat([fully_balanced,tmp[tmp["pdg"].abs()==p].sample(n=lastBest)])
            fully_balanced.to_hdf(full_name,"data",append=False,complevel=0)
            fully_balanced = 0

            if args.partial and not os.path.isfile(bal_name):
                tmp_sig = tmp[selection]
                tmp_bkg = tmp[~selection].sample(n=secondBest)
                tmp = 0
                tmp = pd.concat([tmp_sig,tmp_bkg])
                tmp_sig = 0
                tmp_bkg = 0
                tmp.to_hdf(bal_name,"data",append=False,complevel=0)#,format='table')
            print("Time : %d s" % (time()-t))
            tmp = 0


    else:
            print ("Already Exists")
