import pandas as pd
import os

from tqdm import tqdm

import sys


cols = ["eveNumber",
"runNumber",
"lumNumber",
"puNumInt"]

track_labs = ["pt",
"eta",
"phi",
"p",
"normalizedChi2",
"numberOfValidHits",
"d0",
"dz"] + ["nPixelBarrelHits",
"nPixelEndcapHits",
"nStripTIBHits",
"nStripTOBHits",
"nStripTIDHits",
"nStripTECHits"]

cols = cols + track_labs

cols = cols + ["trackPdg",
"sharedFraction",
"trackMomPdg",
"sharedMomFraction"]
xs = []
ys = []
zs = []
qs = []
for i in range(73):
    xs = xs + ["x_"+str(i)]
    ys = ys + ["y_"+str(i)]
    zs = zs + ["z_"+str(i)]
    rs = rs + ["r_"+str(i)]
    qs = qs + ["q_"+str(i)]

for i in range(73):
    cols = cols + ["x_"+str(i),"y_"+str(i),"z_"+str(i),"phi_"+str(i),
                   "r_"+str(i),"q_"+str(i),"n_"+str(i),"pdg_"+str(i),
                   "mom_"+str(i),"dZ_"+str(i),"ax1_"+str(i),"ax2_"+str(i),
                   "ax3_"+str(i),"ax4_"+str(i),"rawId_"+str(i)]
for i in range(10):
    cols = cols + ["p_size_"+str(i),"p_sizex_"+str(i),"p_sizey_"+str(i),"p_x_"+str(i),
                   "p_y_"+str(i),"p_ovx_"+str(i),"p_ovy_"+str(i),"p_skew_"+str(i),
                   "p_big_"+str(i),"p_bad_"+str(i),"p_edge_"+str(i),"p_charge_"+str(i)]
for i in range(63):
    cols = cols + ["s_dim_"+str(i),"s_center_"+str(i),"s_first_"+str(i),"s_merged_"+str(i),
                   "s_size_"+str(i),"s_charge_"+str(i)]
for i in range(10):
    for j in range(256):
        cols = cols + ["pixel_h_" + str(i) + "_n_"+str(j)]

for i in range(63):
    for j in range(16):
        cols = cols + ["strip_h_" + str(i) + "_n_"+str(j)]

cols = cols + ["dummy"]
dummy = -0.000053421269;

path = "./"
data_files = [path + "/" + f for f in os.listdir(path) if f.endswith(".txt")]
for d in data_files:

    name = d[:-4]
    df = pd.read_table("/lustre/home/adrianodif/CNNTracks/el7/QCD_ML_Crab/0/1_1_1generalTracks_CNN.txt",names=cols)
    df.to_hdf(name + ".h5","data",complevel=0)
    qq = ev_df[qs].values
    xx = ev_df[xs].values
    zz = ev_df[ys].values
    yy = ev_df[zs].values
    rr = ev_df[rs].values

    cut = (xx!=dummy)
    qq = qq[cut]
    xx = xx[cut]
    yy = yy[cut]
    zz = zz[cut]
    rr = rr[cut]
    dd = np.sqrt(xx**2 + yy**2 + zz**2)

    H,_,_ = np.histogram2d(zz,dd,bins=(500,200),range=((-250,250),(0,200)),weights=qq)
    hist_name = name + "_zdw"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,rr,bins=(500,120),range=((-250,250),(0,120)),weights=qq)
    hist_name = name + "_zrw"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,xx,bins=(500,220),range=((-250,250),(-110,110)),weights=qq)
    hist_name = name + "_zxw"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,yy,bins=(500,220),range=((-250,250),(-110,110)),weights=qq)
    hist_name = name + "_zyw"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,dd,bins=(500,200),range=((-250,250),(0,200)))
    hist_name = name + "_zd"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,rr,bins=(500,120),range=((-250,250),(0,120)))
    hist_name = name + "_zr"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,xx,bins=(500,220),range=((-250,250),(-110,110)))
    hist_name = name + "_zx"
    H.tofile(hist_name)

    H,_,_ = np.histogram2d(zz,yy,bins=(500,220),range=((-250,250),(-110,110)))
    hist_name = name + "_zy"
    H.tofile(hist_name)
