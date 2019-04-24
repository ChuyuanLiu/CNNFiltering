# CNNPixelSeeds
Analyzer for CMS Open Data Pixel Seeds ML Applications http://opendata.cern.ch/

Data producer for developing machine learning algorithms to select and filter pixel doublet seeds for tracking applications at CMS experiments.

# Setup

The first step is the creation of a ``CMSSW_10_3_5`` release workarea


``` bash
cmsrel CMSSW_10_3_5
cd CMSSW_10_3_5/src/
git clone git@github.com:cms-legacydata-analyses/CNNPixelSeedsProducerTool.git .
scram b -j 2
cmsenv
```

# Create  `` HDF `` files

cd CNNFiltering/CNNAnalyze/test/


![iterativeTracking](https://raw.githubusercontent.com/AdrianoDee/CNNFiltering/open/iterativeTracking.png)

