import os
from dataset import Dataset
import argparse

parser = argparse.ArgumentParser(prog="balanceHDF")
parser.add_argument('--read', type=str, default="./",help='files path')

args = parser.parse_args()

#remote_data = "/lustre/cms/store/user/adiflori/ConvPixels/TTBar_13TeV_35_PU/allHDFDatasets/"
#remote_data = "/lustre/cms/store/user/adiflori/ConvPixels/TTBar_13TeV_35_PU/0512_724_runs/DataFiles/"
#remote_data = "/lustre/cms/store/user/adiflori/ConvPixels/TTBar_13TeV_35_PU/"
#remote_data = "/lustre/cms/store/user/adiflori/ConvPixels/TTBar_13TeV_35_PU/sept17_runs/"
remote_data = args.read + "/"
#remote_data = "data/inference/unzip/"
new_dir = remote_data + "/bal_data/"

print("Balancing dataset in : " + remote_data)
print("Saving balanced in   : " + new_dir)

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for f in os.listdir(remote_data):
    if os.path.isfile(new_dir + f):  # balancing already done
        print("Skipping (already done): " + f)
        continue
    try:
        ext = f.split('.')[-1]
        print(ext)
        if ext == 'h5':
            print("Loading: " + f)
            train_data = Dataset([remote_data + f]).balance_data()
            train_data.save(new_dir + f)
        elif ext == 'gz':
            print("Loading: " + f)
            with open(remote_data + f, 'rb') as f_zip:
                fc = f_zip.read()
                with open(remote_data + 'tmp/tmp.h5', 'wb') as f_new:
                    f_new.write(fc)
            train_data = Dataset([remote_data + 'tmp/tmp.h5']).balance_data()
            train_data.save(new_dir + f[:-3])  # skip .gz from name
        else:  # balancing already done
            print("Skipping (unrecognized extension): " + f)
    except:
        print("Error loading: " + f)
