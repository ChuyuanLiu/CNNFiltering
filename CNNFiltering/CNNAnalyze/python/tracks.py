# flake8: noqa: E402, F401
#import socket
#if socket.gethostname() == 'cmg-gpu1080':
#    print('locking only one GPU.')
#    import setGPU

import numpy as np
import pandas as pd
import gzip
#from keras.utils.np_utils import to_categorical

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical,num_classes

pt_log_bins = np.logspace(-0.3, 3, 80)

padshape = 16

track_lab = "PdgId"

pdg_lab = "inTpPdgId"

one_pix = "hit_0_IsBarrel"

headLab = ["run","evt","lumi","bSX","bSY","bSZ","bSdZ"]

particle = ["TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                "VX","VY","VZ","DXY","DZ","BunchCrossing","IsChargeMatched",
                "IsSigSimMatched","SharedFraction","NumAssocRecoTracks"]



hitCoord = ["X","Y","Z","Phi","R"] #5

hitDet = ["IsBarrel","Layer","Ladder","Side","Disk","Panel","Module"]#,"IsFlipped","Ax1","Ax2"] #12

hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
            "AvgCharge","OverFlowX","OverFlowY","Skew","IsBig","IsBad","IsEdge"] #13

hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

hitCharge = ["SumADC"]

hitLabs = hitCoord + hitDet + hitClust + hitPixel + hitCharge
hitsLabs = []

for j in range(10):
    hitsLabs.append(["hit_" + str(j) + "_" + str(i) for i in hitLabs])

hitPixels = []
for j in range(10):
    hitPixels.append(["hit_" + str(j) + "_" + str(i) for i in  hitPixel])

hitCoords = []
for j in range(10):
    hitCoords.append(["hit_" + str(j) + "_" + str(i) for i in  hitCoord])

hitFeatures = hitCoord + hitDet + hitClust + hitCharge # 5 + 12 + 13 + 1 = 31

hitsFeatures = []

for j in range(10):
    hitsFeatures.append(["hit_" + str(j) + "_" + str(i) for i in hitFeatures])

hitDets = []

for j in range(10):
    hitDets.append(["hit_" + str(j) + "_" + str(i) for i in hitDet])


featureLabs = []

for j in range(10):
    featureLabs = featureLabs + hitsFeatures[j]

#featureLabs = inHitFeature + outHitFeature + differences

dataLab = headLab

dataLab += particle

for j in range(10):
    dataLab += hitsLabs[j]

dataLab += ["dummyFlag"]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [211., 321. ] #11.,13.,211.,321.,2212.]


class Tracks:
    """ Load the Tracks from files. """

    def __init__(self, fnames,pdgIds=main_pdgs):
        self.data = pd.DataFrame(data=[], columns=dataLab)
        for i,f in enumerate(fnames):
            print("Loading file " + str(i+1) + "/" + str(len(fnames)) + " : " + f)
            df = 0
            if f.lower().endswith("h5"):
                df = pd.read_hdf(f, mode='r')


            if f.lower().endswith(("txt")) or f.lower().endswith(("gz")):
                with open(f, 'rb') as df:
                    #print("Reading file no." + str(no+1) + ": " + d)
                    if f.lower().endswith(("txt")):
                        df = pd.read_table(df, sep="\t", header = None)
                    if f.lower().endswith(("gz")):
                        df = pd.read_table(df, sep="\t", header = None,compression="gzip")

            df.columns = dataLab  # change wrong columns names
            #print(df.head())
            #print(pdgIds)
            #df = data_by_pdg(df,pdgIds)

            df.sample(frac=1.0)
            self.data = self.data.append(df)

    def clean_dataset(self, pdgIds=main_pdgs,verbose=True):
        """ Cleaning: tracks with at least 1 pixel hit. """

        self.data = self.data[self.data[one_pix] >= 0.0]

    def data_by_pdg(self, pdgIds=main_pdgs,verbose=True):
        """ Balancing Tracks by particles. """

        data_pdg = []
        minimum = 1E32
        for p in pdgIds:
            this_pdg = self.data[self.data[target_lab].abs() == p]
            data_pdg.append(this_pdg)
            minimum = min(minimum,this_pdg.shape[0])
            if verbose:
                print("Number of " + str(p) + " : " + str(this_pdg.shape[0]))

        data_pdg_bal = []
        for d in data_pdg:
            if d.shape[0] > minimum:
                data_pdg_bal.append(d.sample(minimum))
            else:
                data_pdg_bal.append(d)

        data_tot = pd.concat(data_pdg_bal)
        data_tot = data_tot.sample(frac=1.0)

        if verbose:
            for p,d in zip(pdgIds,data_pdg_bal):
                print(" - New no. of " + str(p) + " : " + str(d.shape[0]))

        self.data = data_tot

    def data_by_pt(self,verbose=True):
        """ Balancing Tracks by pts. """

        data_pts = []
        minimum = 1E32
        for i in range(len(pt_log_bins)-1):
            this_pt = self.data[(self.data["Pt"] >= pt_log_bins[i]) & (self.data["Pt"] <= pt_log_bins[i+1])]
            data_pts.append(this_pt)
            thres = max(this_pt.shape[0],1)
            minimum = int(min(minimum,int(5.0 * thres)))
            if verbose:
                print("Number of tracks in pt range [" + str(pt_log_bins[i]) + "," + str(pt_log_bins[i+1]) + "] : " + str(this_pt.shape[0]))

        print(minimum)
        data_pts_bal = []
        for d in data_pts:

            if d.shape[0] > minimum:
                data_pts_bal.append(d.sample(minimum))
            else:
                data_pts_bal.append(d)

        data_tot = pd.concat(data_pts_bal)
        data_tot = data_tot.sample(frac=1.0)

        if verbose:
            for i in range(len(pt_log_bins)-1):
                this_pt = data_tot[(data_tot["Pt"] >= pt_log_bins[i]) & (data_tot["Pt"] <= pt_log_bins[i+1])]
                print("Number of tracks in pt range [" + str(pt_log_bins[i]) + "," + str(pt_log_bins[i+1]) + "] : " + str(this_pt.shape[0]))

        print(self.data.shape[0])
        self.data = data_tot

    def from_dataframe(self,data):
        """ Constructor method to initialize the classe from a DataFrame """
        self.data = data


    def recolumn(self):
        self.data.columns = dataLab

    def b_w_correction(self, hit,smoothing=1.0):

        turned  = ((hit > 0.).astype(float)) * smoothing

        return turned


    def filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        self.data = self.data[self.data[feature_name] == value]
        return self  # to allow method chaining

    def Filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        d = Tracks(self.data[self.data[feature_name] == value])

	d.data =  self.data[self.data[feature_name] == value]
        return d  # to allow method chaining

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featureLabs].as_matrix()

    def get_track_hits_data(self,bw=False):

        self.recolumn()

        #self.data_augmentation(magnitude=augmentation)
        hits = []
        mean, std = (13382.0011321,10525.1252954)

        pixelHits = []

        for i in range(10):
            h = self.data[hitPixels[i]].as_matrix().astype(np.float16)
            #print h
            h = (h - mean) / std
            if bw:
                h = self.b_w_correction(h)

            pixelHits.append(h)

        data = np.array(pixelHits)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data),-1,padshape, padshape))
        X_track = np.transpose(data, (1, 2, 3, 0))

        #print(X_track[0,:,:,0])

        X_info = self.get_info_features()
        y,_= to_categorical(self.get_track_labels())

        return X_track, X_info, y


    def get_track_labels(self,pdgIds=main_pdgs):
        labels = np.full(len(self.data[target_lab].as_matrix()),0.0)
        for p in pdgIds:
            labels[(self.data[target_lab].abs().as_matrix()==p)] = main_pdgs.index(p)

        print set(labels)
        return labels

    def save(self, fname):
        # np.save(fname, self.data.as_matrix())
        self.data.to_hdf(fname, 'data', mode='w')

if __name__ == '__main__':
    d = Tracks('data/debug.npy')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_data()
    assert x[0].shape == (batch_size, padshape, padshape, 8)

    x = d.get_data(normalize=False, angular_correction=False,
                   flipped_channels=False)[0]
    assert x.shape == (batch_size, padshape, padshape, 2)
    np.testing.assert_allclose(
        x[:, :, :, 0], d.data[inPixels].as_matrix().reshape((-1, padshape, padshape)))

    print("All test successfully completed.")
