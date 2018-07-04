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

padshape = 15

target_lab = ["true"]

headLab  = ["event_id"]
hitCoord = ["hit_id","x","y","z","phi","r"]
hitClust = ["avg0","min0","max0","avg1","min1","max1","size0","size1","value"]
hitDet   = ["volume_id","layer_id","module_id"]
hitPixel = ["Pix" + str(el) for el in range(0, padshape*padshape)]

inPixels  = [ "in_" + str(i) for i in hitPixel]
outPixels = [ "out_" + str(i) for i in hitPixel]

allPixels = inPixels + outPixels

hitLabs = headLab + hitCoord + hitDet + hitClust + hitPixel

inHitLabs = [ "in_" + str(i) for i in hitLabs]
outHitLabs = [ "out_" + str(i) for i in hitLabs]

hitFeatures = hitCoord + hitClust # 5 + 12 + 13 + 1 = 31

inHitFeature  = [ "in_" + str(i) for i in hitFeatures]
outHitFeature = [ "out_" + str(i) for i in hitFeatures]

particleLabs = ["particle_id","pt","px","py","pz"]
inParticleLabs  = [ "in_" + str(i) for i in particleLabs]
outParticleLabs = [ "out_" + str(i) for i in particleLabs]

differences = ["deltaR", "deltaPhi", "deltaZ"]

featureLabs = inHitFeature + outHitFeature + differences

dataLab = inHitLabs + outHitLabs + differences + inParticleLabs + outParticleLabs + target_lab + ["key"] + ["inDet","outDet"]


layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]


class Dataset:
    """ Load the dataset from txt files. """

    def __init__(self, fnames,balance=False,pdgIds=main_pdgs):
        self.data = pd.DataFrame(data=[], columns=dataLab)
        for i,f in enumerate(fnames):
            print("Loading file " + str(i+1) + "/" + str(len(fnames)) + " : " + f)
            df = 0
            if not f.lower().endswith("h5"):
                continue

            df = pd.read_hdf(f, mode='r')
            df = df[dataLab]
            if balance:
                df = balance_data_by_pdg(df,pdgIds)
            df.columns = dataLab  # change wrong columns names
            df.sample(frac=1.0)
            self.data = self.data.append(df)

    def from_dataframe(self,data):
        """ Constructor method to initialize the classe from a DataFrame """
        self.data = data

    def recolumn(self):
        self.data.columns = dataLab

    def b_w_correction(self, hits_in, hits_out,smoothing=1.0):

        self.recolumn()
        turned_in  = ((hits_in > 0.).astype(float)) * smoothing
        turned_out = ((hits_out > 0.).astype(float)) * smoothing

        return turned_in,turned_out


    def get_hit_shapes(self, normalize=True, angular_correction=True, flipped_channels=True, bw_cluster = True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inPixels].as_matrix()
        a_out = self.data[outPixels].as_matrix()
        self.recolumn()
        # Normalize data
        if normalize:
            mean, std = (13382.0011321,10525.1252954) #on 2.5M hits PU35
            a_in = a_in / std
            a_out = a_out / std

        if bw_cluster:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        if flipped_channels:
            flip_in, not_flip_in = self.separate_flipped_hits(
                a_in, self.data.isFlippedIn)
            flip_out, not_flip_out = self.separate_flipped_hits(
                a_out, self.data.isFlippedOut)
            l = [flip_in, not_flip_in, flip_out, not_flip_out]
        else:
            l = [a_in, a_out]

        if angular_correction:
            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
                a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]
            phic_in, phic_out, phis_in, phis_out = self.phi_correction(
                a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        # TODO: not optimal for CPU execution
        return np.transpose(data, (1, 2, 3, 0))

    def get_hit_dense(self, normalize=True, angular_correction=True, flipped_channels=True, bw_cluster = True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inPixels].as_matrix()
        a_out = self.data[outPixels].as_matrix()
        self.recolumn()
        # Normalize data
        
        if normalize:
            mean, std = (13382.0011321,10525.1252954) #on 2.5M hits PU35
            a_in = a_in / std
            a_out = a_out / std

        if bw_cluster:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        if flipped_channels:
            flip_in, not_flip_in = self.separate_flipped_hits(
                a_in, self.data.isFlippedIn)
            flip_out, not_flip_out = self.separate_flipped_hits(
                a_out, self.data.isFlippedOut)
            l = [flip_in, not_flip_in, flip_out, not_flip_out]
        else:
            l = [a_in, a_out]

        if angular_correction:
            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
                a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]
            phic_in, phic_out, phis_in, phis_out = self.phi_correction(
                a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        # TODO: not optimal for CPU execution
        return data

    def filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        self.data = self.data[self.data[feature_name] == value]
        return self  # to allow method chaining

    def Filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        d = Dataset(self.data[self.data[feature_name] == value])

        d.data =  self.data[self.data[feature_name] == value]
        return d  # to allow method chaining

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featureLabs].as_matrix()

    def get_layer_map_data(self,theta=False,phi=False,bw=False):

        a_in = self.data[inPixels].as_matrix().astype(np.float16)
        a_out = self.data[outPixels].as_matrix().astype(np.float16)
        
        pix = self.data[allPixels].values.flatten()
        pix = pix[pix!=0.0]
     
        mean, std = (np.mean(pix),np.std(pix)) #on 2.5M doublets
       
        a_in[a_in!=0.0]   = (a_in[a_in!=0.0] - mean) / std
        a_out[a_out!=0.0] = (a_out[a_out!=0.0] - mean) / std

        if bw:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        l = []

        if theta:

            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]

        if phi:

            phic_in, phic_out, phis_in, phis_out = self.phi_correction(a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        for hits, ids in [(a_in, self.data.inDet), (a_out, self.data.outDet)]:

            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        X_hit = np.transpose(data, (1, 2, 3, 0))

        #print(X_hit[0,:,:,0])

        X_info = self.get_info_features()
        y,_= to_categorical(self.get_labels())

        return X_hit, X_info, y


    def get_labels(self):
        return self.data[target_lab].as_matrix() == 0.0

    def get_labels_multiclass(self):
        labels = np.full(len(self.data[target_lab].as_matrix()),1.0)
        labels[self.data[target_lab].as_matrix()==-1.0] = 0.0
        for p in main_pdgs:
            labels[(self.data[pdg_lab].abs().as_matrix()==p) & (self.data[target_lab].as_matrix()!=-1.0)] = main_pdgs.index(p) + 2

        print set(labels)
        return labels

    def get_data(self, normalize=True, angular_correction=True, flipped_channels=True,b_w_correction=False):
        X_hit = self.get_hit_shapes(
            normalize, angular_correction, flipped_channels,b_w_correction)
        X_info = self.get_info_features()
        y = to_categorical(self.get_labels(), num_classes=2)
        return X_hit, X_info, y

    def get_data_dense(self, normalize=True, angular_correction=True, flipped_channels=True,b_w_correction=False):
        X_hit = self.get_hit_dense(
            normalize, angular_correction, flipped_channels,b_w_correction)
        X_info = self.get_info_features()

        X = np.hstack((X_hit,X_info))
        y = to_categorical(self.get_labels(), num_classes=2)
        return X, y

    def save(self, fname):
        # np.save(fname, self.data.as_matrix())
        self.data.to_hdf(fname, 'data', mode='w')

    # TODO: pick doublets from same event.
    def balance_data(self, max_ratio=0.5, verbose=True):
        """ Balance the data. """
        data_neg = self.data[self.data[target_lab] != 0.0]
        data_pos = self.data[self.data[target_lab] == 0.0]

        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]

	if n_pos==0:
		print("Number of negatives: " + str(n_neg))
                print("Number of positive: " + str(n_pos))
 		print("Returning")
		return self
        if verbose:
            print("Number of negatives: " + str(n_neg))
            print("Number of positive: " + str(n_pos))
            print("Ratio: " + str(n_neg / n_pos))

        if n_pos > n_neg:
            return self

        data_neg = data_neg.sample(n_pos)
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
        self.data = balanced_data
        return self  # allow method chaining


class DataGenerator:

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=20,n_classes=2, shuffle=True):
    	"""Generator Definition"""
	self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
    	'''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
    	'''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
    	'''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    	'''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

if __name__ == '__main__':
    d = Dataset('data/debug.npy')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_data()
    assert x[0].shape == (batch_size, padshape, padshape, 8)

    x = d.get_data(normalize=False, angular_correction=False,
                   flipped_channels=False)[0]
    assert x.shape == (batch_size, padshape, padshape, 2)
    np.testing.assert_allclose(
        x[:, :, :, 0], d.data[inPixels].as_matrix().reshape((-1, padshape, padshape)))

    print("All test successfully completed.")
