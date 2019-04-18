import pandas as pd
import os

import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',  type=str, default="./")
#parser.add_argument('--split', type=int, default=-1)
#parser.add_argument('--dir',   type=str, default=None)
#parser.add_argument('--tree',  type=str, default="2mu2kSkimmedTree")
args = parser.parse_args()


data_files = [data_path + f for f in os.listdir(args.path) if f.endswith(".h5")]

print("Balancing")

for ff in data_files:
    if not os.path.isfile(ff[:-3]+"_bal.h5"):
            name = ff[:-3]

            print("Loading: " + name)
            tmp = pd.read_hdf(ff)
            print("Loading File : " + f.split("/")[-1])
            bal_name = f[:-3] + "_bal.h5"

            t = time()
            tmp = 0
            tmp = pd.read_hdf(f)

            #uproot.open(cnn_file)["TrackProducer"]["CnnTracks"].pandas.df()
            secondBest = (tmp["pdg"].value_counts().values)[1]
            selection = (tmp["pdg"]>-9000.0)

            tmp_sig = tmp[selection]
            tmp_bkg = tmp[~selection].sample(n=secondBest)

            tmp = 0
            tmp = pd.concat([tmp_sig,tmp_bkg])
            tmp_sig = 0
            tmp_bkg = 0
            tmp.to_hdf(bal_name,"data",complevel=0,append=False,complevel=0,format='table')
            print("Time : %d s" % (time()-t))
            tmp = 0
    else:
            print ("Already Exists")
