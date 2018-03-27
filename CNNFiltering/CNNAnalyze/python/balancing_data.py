import os
from dataset import Dataset
import argparse

parser = argparse.ArgumentParser(prog="detectorHDF")
parser.add_argument('--read', type=str, default="./",help='files path')
parser.add_argument('--chunk', type=int, default="50",help='chunk size')
parser.add_argument('--offset', type=int, default="0",help='offset size')
parser.add_argument('--limit', type=int, default="10000",help='offset size')
parser.add_argument('-d','--debug',action='store_true')
args = parser.parse_args()

remote_data = args.read + "/"
chunksize   = args.chunk
offset      = args.offset
#remote_data = "data/inference/unzip/"
new_dir = remote_data + "/det_data/"

files = [remote_data + el for el in os.listdir(remote_data)]

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for chunk in  range(offset,int(((len(files) + chunksize))/chunksize) + 1):
    if(min(len(files),chunk*chunksize)==min(len(files),chunksize*(chunk+1)) and chunk!=0):
        continue
        
    if chunk > offset + limit:
        break

    if args.debug:
        p = files[:2]
    else:
        p = files[min(len(files),chunk*chunksize):min(len(files),chunksize*(chunk+1))]

    print("loading & balancing data...")
    data = Dataset(p).balance_by_det().balance_by_pdg()
    print("dumping data...")
    data.save(remote_data + "/pdg_bal_dataset_" + str(chunk) + ".h5")
