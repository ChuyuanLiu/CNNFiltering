import numpy as np
import pandas as pd
import gzip

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

padshape = 16

target_lab = "label"

pdg_lab = "inTpPdgId"

headLab = ["run","evt","lumi","PU","detSeqIn","detSeqOut","bSX","bSY","bSZ","bSdZ"]

hitCoord = ["X","Y","Z","Phi","R"] #5

hitDet = ["DetSeq","IsBarrel","Layer","Ladder","Side","Disk","Panel","Module","IsFlipped","Ax1","Ax2"] #12

hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
            "AvgCharge","OverFlowX","OverFlowY","Skew","IsBig","IsBad","IsEdge"] #13

hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

hitCharge = ["SumADC"]

hitLabs = hitCoord + hitDet + hitClust + hitPixel + hitCharge

inHitLabs = [ "in" + str(i) for i in hitLabs]
outHitLabs = [ "out" + str(i) for i in hitLabs]

inPixels = [ "in" + str(i) for i in hitPixel]
outPixels = [ "out" + str(i) for i in hitPixel]


labels = ["PId","TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                "VX","VY","VZ","DXY","DZ","BunchCrossing","IsChargeMatched",
                "IsSigSimMatched","SharedFraction","NumAssocRecoTracks"]

track = ["Track_" + el]
particle = ["PId","TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                "VX","VY","VZ","DXY","DZ","BunchCrossing","IsChargeMatched",
                "IsSigSimMatched","SharedFraction","NumAssocRecoTracks"]

hitFeatures = hitCoord + hitDet + hitClust + hitCharge # 5 + 12 + 13 + 1 = 31

inParticle = [ "inTp" + str(i) for i in particle]
outParticle = [ "outTp" + str(i) for i in particle]

inHitFeature  = [ "in" + str(i) for i in hitFeatures]
outHitFeature = [ "out" + str(i) for i in hitFeatures]

particleLabs = ["label","intersect","particles"] + inParticle +  outParticle

differences = ["deltaA", "deltaADC", "deltaS", "deltaR", "deltaPhi", "deltaZ", "ZZero"]

featureLabs = inHitFeature + outHitFeature + differences

dataLab = headLab + inHitLabs + outHitLabs + differences + particleLabs + ["dummyFlag"]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

def loadTpTracks(fnames):
    data = pd.DataFrame(data=[], columns=dataLab)
    for i,f in enumerate(fnames):
        print("Loading file " + str(i+1) + "/" + str(len(fnames)) + " : " + f)
        df = 0
        if not f.lower().endswith("h5"):
            continue

        df = pd.read_hdf(f, mode='r')

        df.columns = dataLab  # change wrong columns names
        df.sample(frac=1.0)
        data = self.data.append(df)

    data.to_hdf(fname, 'data', mode='w')
