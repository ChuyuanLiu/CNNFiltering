# CNNPixelSeeds
Analyzer for CMS Open Data Pixel Seeds ML Applications http://opendata.cern.ch/

Data producer for developing machine learning algorithms to select and filter pixel doublet seeds for tracking applications at CMS experiments.

### Setup

The first step is the creation of a ``CMSSW_10_2_5`` release workarea

``` bash
cmsrel CMSSW_10_2_5
cd CMSSW_10_2_5/src/
git clone git@github.com:cms-legacydata-analyses/CNNPixelSeedsProducerTool.git .
scram b -j 2
cmsenv
```

### Dumping doublets in  ` txt ` files

Once the compilation is completed you are ready to produce the pixel doublets seeds datasets:
 
``` bash
cmsrel CMSSW_10_3_5
cd CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/
cmsRun step3_ML_trackingOnly.py
```

This configuration will run the full CMS track reconstruction on simulated (![equation](http://latex.codecogs.com/gif.latex?t\bar{t})) events and will produce in `CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/doublets/` directory (automatically created) a set of text (`txt`) files containing the doublets produced in each of the pixel seed based iterative tracking steps (red boxed in the picture below). For further details about track reconstruction at CMS and iterative tracking see [1],[2],[3]

![iterativeTracking](https://raw.githubusercontent.com/AdrianoDee/CNNFiltering/open/iterativeTracking.png)

Analogously 

``` bash
cmsrel CMSSW_10_3_5
cd CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/
cmsRun step3_ML_pixelOnly.py
```

will produce the doublets seeds used aas starting blocks for pixel-only tracks reconstruction. The text files generated are named with the following rules:

`_l_r_e_step_dnn_doublets.txt`

with

* l = lumisection number
* r = run number
* e = event number
* step = iterative tracking step name (e.g `pixelTracksHitDoublets` for pixel tracks step)

Both the configuration files (`step3_ML_pixelOnly.py` and `step3_ML_trackingOnly.py`) may receive in input few parameters:

| Name       | Type                 | Default | Description                                                                                    |
|------------|----------------------|---------|------------------------------------------------------------------------------------------------|
| pileUp     | int                  | 50      | Average number of simultaneous collisions per event  (for this use case should be kept to 50). |
| skipEvent  | int                  | 0       | Number of events to be skipped.                                                                |
| numEvents  | int                  | 100     | Total number of events to be processed (after skipping).                                       |
| numFile    | int                  | 0       | The index, in the list provided, of the file to be processed.                                  |
| openDataVM | bool (True or False) | True    | Flag to signal if you are working on an Open Data WM or somewhere else.                        
|

Any of these inputs should be parsed as follows:

`cmsRun step3_ML_trackingOnly.py inputName=VALUE`



### Conversion to ` HDF ` files

In order to convert the txt datasets to hdf table formats simply run (in `CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/`)

` python toHdf.py`

this will automatically read the content of `doublets` directory and produce two hdf files:

* in `doublets/original/` the plain hdf converted file;
* in `doublets/bal_data/` a new balanced hdf table where the yields of fake and true seeds have been forced to be equal, by sampling the more populated of the two classes;

### The dataset

The dataset created above consists of a collection of pixel doublet seeds that would be used by CMS track reconstruction workflow. Each doublet is characterised by a list of features:



The notebook in `CNNPixelSeedsProducerTool/notebooks/cnn_filtering.ipynb` is a good starting point to explore and understand the datset features.


[1] https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideIterativeTracking

[2] https://cds.cern.ch/record/2308020

[3] https://cds.cern.ch/record/2293435
