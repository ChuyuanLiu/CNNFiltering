#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "HitExtractorPIX.h"
#include "HitExtractorSTRP.h"

#include <iostream>
#include <sstream>
#include <ostream>
#include <fstream>
#include <map>


using namespace ctfseeding;
using namespace std;

SeedingLayerSetsBuilder::SeedingLayerId SeedingLayerSetsBuilder::nameToEnumId(const std::string& name) {
    GeomDetEnumerators::SubDetector subdet = GeomDetEnumerators::invalidDet;
    TrackerDetSide side = TrackerDetSide::Barrel;
    int idLayer = 0;

    size_t index;
    //
    // BPIX
    //
    if ((index = name.find("BPix")) != string::npos) {
      subdet = GeomDetEnumerators::PixelBarrel;
      side = TrackerDetSide::Barrel;
      idLayer = atoi(name.substr(index+4,1).c_str());
    }
    //
    // FPIX
    //
    else if ((index = name.find("FPix")) != string::npos) {
      subdet = GeomDetEnumerators::PixelEndcap;
      idLayer = atoi(name.substr(index+4).c_str());
      if ( name.find("pos") != string::npos ) {
        side = TrackerDetSide::PosEndcap;
      } else {
        side = TrackerDetSide::NegEndcap;
      }
    }
    //
    // TIB
    //
    else if ((index = name.find("TIB")) != string::npos) {
      subdet = GeomDetEnumerators::TIB;
      side = TrackerDetSide::Barrel;
      idLayer = atoi(name.substr(index+3,1).c_str());
    }
    //
    // TID
    //
    else if ((index = name.find("TID")) != string::npos) {
      subdet = GeomDetEnumerators::TID;
      idLayer = atoi(name.substr(index+3,1).c_str());
      if ( name.find("pos") != string::npos ) {
        side = TrackerDetSide::PosEndcap;
      } else {
        side = TrackerDetSide::NegEndcap;
      }
    }
    //
    // TOB
    //
    else if ((index = name.find("TOB")) != string::npos) {
      subdet = GeomDetEnumerators::TOB;
      side = TrackerDetSide::Barrel;
      idLayer = atoi(name.substr(index+3,1).c_str());
    }
    //
    // TEC
    //
    else if ((index = name.find("TEC")) != string::npos) {
      subdet = GeomDetEnumerators::TEC;
      idLayer = atoi(name.substr(index+3,1).c_str());
      if ( name.find("pos") != string::npos ) {
        side = TrackerDetSide::PosEndcap;
      } else {
        side = TrackerDetSide::NegEndcap;
      }
    }
    return std::make_tuple(subdet, side, idLayer);
}

SeedingLayerSetsBuilder::LayerSpec::LayerSpec(unsigned short index, const std::string& layerName, const edm::ParameterSet& cfgLayer, edm::ConsumesCollector& iC):
  nameIndex(index),
  hitBuilder(cfgLayer.getParameter<string>("TTRHBuilder"))
{
  usePixelHitProducer = false;
  if (cfgLayer.exists("HitProducer")) {
    pixelHitProducer = cfgLayer.getParameter<string>("HitProducer");
    usePixelHitProducer = true;
  }

  bool skipClusters = cfgLayer.exists("skipClusters");
  if (skipClusters) {
    LogDebug("SeedingLayerSetsBuilder")<<layerName<<" ready for skipping";
  }
  else{
    LogDebug("SeedingLayerSetsBuilder")<<layerName<<" not skipping ";
  }

  auto subdetData = nameToEnumId(layerName);
  subdet = std::get<0>(subdetData);
  side = std::get<1>(subdetData);
  idLayer = std::get<2>(subdetData);
  if(subdet == GeomDetEnumerators::PixelBarrel ||
     subdet == GeomDetEnumerators::PixelEndcap) {
    extractor = std::make_unique<HitExtractorPIX>(side, idLayer, pixelHitProducer, iC);
  }
  else if(subdet != GeomDetEnumerators::invalidDet) { // strip
    auto extr = std::make_unique<HitExtractorSTRP>(subdet, side, idLayer, clusterChargeCut(cfgLayer) );
    if (cfgLayer.exists("matchedRecHits")) {
      extr->useMatchedHits(cfgLayer.getParameter<edm::InputTag>("matchedRecHits"), iC);
    }
    if (cfgLayer.exists("rphiRecHits")) {
      extr->useRPhiHits(cfgLayer.getParameter<edm::InputTag>("rphiRecHits"), iC);
    }
    if (cfgLayer.exists("stereoRecHits")) {
      extr->useStereoHits(cfgLayer.getParameter<edm::InputTag>("stereoRecHits"), iC);
    }
    if (cfgLayer.exists("useRingSlector") && cfgLayer.getParameter<bool>("useRingSlector")) {
      extr->useRingSelector(cfgLayer.getParameter<int>("minRing"),
                                 cfgLayer.getParameter<int>("maxRing"));
    }
    bool useSimpleRphiHitsCleaner = cfgLayer.exists("useSimpleRphiHitsCleaner") ? cfgLayer.getParameter<bool>("useSimpleRphiHitsCleaner") : true;
    extr->useSimpleRphiHitsCleaner(useSimpleRphiHitsCleaner);

    double minAbsZ = cfgLayer.exists("MinAbsZ") ? cfgLayer.getParameter<double>("MinAbsZ") : 0.;
    if(minAbsZ > 0.) {
      extr->setMinAbsZ(minAbsZ);
    }
    if(skipClusters) {
      bool useProjection = cfgLayer.exists("useProjection") ? cfgLayer.getParameter<bool>("useProjection") : false;
      if(useProjection) {
        LogDebug("SeedingLayerSetsBuilder")<<layerName<<" will project partially masked matched rechit";
      }
      else {
        extr->setNoProjection();
      }
    }
    extractor = std::move(extr);
  }
  if(extractor && skipClusters) {
    extractor->useSkipClusters(cfgLayer.getParameter<edm::InputTag>("skipClusters"), iC);
  }
}

std::string SeedingLayerSetsBuilder::LayerSpec::print(const std::vector<std::string>& names) const
{
  std::ostringstream str;
  str << "Layer="<<names[nameIndex]<<", hitBldr: "<<hitBuilder;

  str << ", useRingSelector: ";
  HitExtractorSTRP *ext = nullptr;
  if((ext = dynamic_cast<HitExtractorSTRP *>(extractor.get())) &&
     ext->useRingSelector()) {
    auto minMaxRing = ext->getMinMaxRing();
    str <<"true,"<<" Rings: ("<< std::get<0>(minMaxRing) <<","<< std::get<1>(minMaxRing) <<")";
  } else  str<<"false";

  return str.str();
}
//FastSim specific constructor
SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC, const edm::InputTag& fastsimHitTag):
  SeedingLayerSetsBuilder(cfg, iC)
{
  fastSimrecHitsToken_ = iC.consumes<FastTrackerRecHitCollection>(fastsimHitTag);
}
SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector&& iC):
  SeedingLayerSetsBuilder(cfg, iC)
{}
SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC)
{
  std::vector<std::string> namesPset = cfg.getParameter<std::vector<std::string> >("layerList");
  std::vector<std::vector<std::string> > layerNamesInSets = this->layerNamesInSets(namesPset);
  // debug printout of layers
  typedef std::vector<std::string>::const_iterator IS;
  typedef std::vector<std::vector<std::string> >::const_iterator IT;
  std::ostringstream str;
  // The following should not be set to cout
//  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
//    str << "SET: ";
//    for (IS is = it->begin(); is != it->end(); is++)  str << *is <<" ";
//    str << std::endl;
//  }
//  std::cout << str.str() << std::endl;
  if(layerNamesInSets.empty())
    theNumberOfLayersInSet = 0;
  else
    theNumberOfLayersInSet = layerNamesInSets[0].size();


  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
    if(it->size() != theNumberOfLayersInSet)
      throw cms::Exception("Configuration") << "Assuming all SeedingLayerSets to have same number of layers. LayerSet " << (it-layerNamesInSets.begin()) << " has " << it->size() << " while 0th has " << theNumberOfLayersInSet;
    for(const std::string& layerName: *it) {
      auto found = std::find(theLayerNames.begin(), theLayerNames.end(), layerName);
      unsigned short layerIndex = 0;
      if(found != theLayerNames.end()) {
        layerIndex = found-theLayerNames.begin();
      }
      else {
        if(std::numeric_limits<unsigned short>::max() == theLayerNames.size()) {
          throw cms::Exception("Assert") << "Too many layers in " << __FILE__ << ":" << __LINE__ << ", we may have to enlarge the index type from unsigned short to unsigned int";
        }

        layerIndex = theLayers.size();
        theLayers.emplace_back(theLayerNames.size(), layerName, layerConfig(layerName, cfg), iC);
        theLayerNames.push_back(layerName);
      }
      theLayerSetIndices.push_back(layerIndex);
    }
  }
  theLayerDets.resize(theLayers.size());
  theTTRHBuilders.resize(theLayers.size());

  // debug printout
  // The following should not be set to cout
  //for(const LayerSpec& layer: theLayers) {
  //  std::cout << layer.print(theLayerNames) << std::endl;
  //}
}

SeedingLayerSetsBuilder::~SeedingLayerSetsBuilder() {}

void SeedingLayerSetsBuilder::fillDescriptions(edm::ParameterSetDescription& desc) {
  edm::ParameterSetDescription empty;
  empty.setAllowAnything(); // for now accept any parameter in the PSets, consider improving later

  desc.add<std::vector<std::string> >("layerList", {});
  desc.add<edm::ParameterSetDescription>("BPix", empty);
  desc.add<edm::ParameterSetDescription>("FPix", empty);
  desc.add<edm::ParameterSetDescription>("TIB", empty);
  desc.add<edm::ParameterSetDescription>("TID", empty);
  desc.add<edm::ParameterSetDescription>("TOB", empty);
  desc.add<edm::ParameterSetDescription>("TEC", empty);
  desc.add<edm::ParameterSetDescription>("MTIB", empty);
  desc.add<edm::ParameterSetDescription>("MTID", empty);
  desc.add<edm::ParameterSetDescription>("MTOB", empty);
  desc.add<edm::ParameterSetDescription>("MTEC", empty);
}

edm::ParameterSet SeedingLayerSetsBuilder::layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const
{
  edm::ParameterSet result;

  for (string::size_type iEnd=nameLayer.size(); iEnd > 0; --iEnd) {
    string name = nameLayer.substr(0,iEnd);
    if (cfg.exists(name)) return cfg.getParameter<edm::ParameterSet>(name);
  }
  edm::LogError("SeedingLayerSetsBuilder") <<"configuration for layer: "<<nameLayer<<" not found, job will probably crash!";
  return result;
}

vector<vector<string> > SeedingLayerSetsBuilder::layerNamesInSets( const vector<string> & namesPSet)
{
  std::vector<std::vector<std::string> > result;
  for (std::vector<std::string>::const_iterator is=namesPSet.begin(); is < namesPSet.end(); ++is) {
    vector<std::string> layersInSet;
    string line = *is;
    string::size_type pos=0;
    while (pos != string::npos ) {
      pos=line.find("+");
      string layer = line.substr(0,pos);
      layersInSet.push_back(layer);
      line=line.substr(pos+1,string::npos);
    }
    result.push_back(layersInSet);
  }
  return result;
}

void SeedingLayerSetsBuilder::updateEventSetup(const edm::EventSetup& es) {
  // We want to evaluate both in the first invocation (to properly
  // initialize ESWatcher), and this way we avoid one branch compared
  // to || (should be tiny effect)
  if(! (geometryWatcher_.check(es) | trhWatcher_.check(es)) )
    return;

  edm::ESHandle<GeometricSearchTracker> htracker;
  es.get<TrackerRecoGeometryRecord>().get( htracker );
  const GeometricSearchTracker& tracker = *htracker;

  const std::vector<BarrelDetLayer const*>&  bpx  = tracker.barrelLayers();
  const std::vector<BarrelDetLayer const*>&  tib  = tracker.tibLayers();
  const std::vector<BarrelDetLayer const*>&  tob  = tracker.tobLayers();

  const std::vector<ForwardDetLayer const*>& fpx_pos = tracker.posForwardLayers();
  const std::vector<ForwardDetLayer const*>& tid_pos = tracker.posTidLayers();
  const std::vector<ForwardDetLayer const*>& tec_pos = tracker.posTecLayers();

  const std::vector<ForwardDetLayer const*>& fpx_neg = tracker.negForwardLayers();
  const std::vector<ForwardDetLayer const*>& tid_neg = tracker.negTidLayers();
  const std::vector<ForwardDetLayer const*>& tec_neg = tracker.negTecLayers();

  for(const auto& layer: theLayers) {
    const DetLayer * detLayer = nullptr;
    int index = layer.idLayer-1;

    if (layer.subdet == GeomDetEnumerators::PixelBarrel) {
      detLayer = bpx[index];
    }
    else if (layer.subdet == GeomDetEnumerators::PixelEndcap) {
      if (layer.side == TrackerDetSide::PosEndcap) {
        detLayer = fpx_pos[index];
      } else {
        detLayer = fpx_neg[index];
      }
    }
    else if (layer.subdet == GeomDetEnumerators::TIB) {
      detLayer = tib[index];
    }
    else if (layer.subdet == GeomDetEnumerators::TID) {
      if (layer.side == TrackerDetSide::PosEndcap) {
        detLayer = tid_pos[index];
      } else {
        detLayer = tid_neg[index];
      }
    }
    else if (layer.subdet == GeomDetEnumerators::TOB) {
      detLayer = tob[index];
    }
    else if (layer.subdet == GeomDetEnumerators::TEC) {
      if (layer.side == TrackerDetSide::PosEndcap) {
        detLayer = tec_pos[index];
      } else {
        detLayer = tec_neg[index];
      }
    }
    else {
      throw cms::Exception("Configuration") << "Did not find DetLayer for layer " << theLayerNames[layer.nameIndex];
    }

    edm::ESHandle<TransientTrackingRecHitBuilder> builder;
    es.get<TransientRecHitRecord>().get(layer.hitBuilder, builder);

    theLayerDets[layer.nameIndex] = detLayer;
    theTTRHBuilders[layer.nameIndex] = builder.product();
  }
}

std::vector<SeedingLayerSetsBuilder::SeedingLayerId> SeedingLayerSetsBuilder::layers() const {
  std::vector<SeedingLayerId> ret;
  ret.reserve(numberOfLayers());
  for(const auto& layer: theLayers) {
    ret.emplace_back(layer.subdet, layer.side, layer.idLayer);
  }
  return ret;
}

std::unique_ptr<SeedingLayerSetsHits> SeedingLayerSetsBuilder::hits(const edm::Event& ev, const edm::EventSetup& es) {
  updateEventSetup(es);

  auto ret = std::make_unique<SeedingLayerSetsHits>(theNumberOfLayersInSet,
                                                    &theLayerSetIndices,
                                                    &theLayerNames,
                                                    &theLayerDets);

  int eveNumber = ev.id().event();
  int runNumber = ev.id().run();
  int lumNumber = ev.id().luminosityBlock();

  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
  tpMap_ = consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap"));

  edm::Handle<ClusterTPAssociation> tpClust;
  iEvent.getByToken(tpMap_,tpClust);

  for(auto& layer: theLayers) {

    std::cout << "Layer " << layer.subdet << " - " << int(layer.side) << " - " << layer.idLayer << std::endl;
    std::string fileName = "hits/" + std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
    fileName += "_" + std::to_string(layer.subdet) + "_" + std::to_string(int(layer.side)) + "_" + std::to_string(layer.idLayer) + "_hits.txt";
    std::ofstream outHitFile(fileName, std::ofstream::app);


    auto theHits = layer.extractor->hits((const TkTransientTrackingRecHitBuilder &)(*theTTRHBuilders[layer.nameIndex]), ev, es);
    std::cout << "The hits size: " << theHits.size() << std::endl;

    const SiPixelRecHit*   theHitPix = dynamic_cast<const SiPixelRecHit* >(&(*theHits[0]));
    const SiStripRecHit1D* theHitsStrip = dynamic_cast<const SiStripRecHit1D* >(&(*theHits[0]));
    const SiStripRecHit2D* theHitsStrip = dynamic_cast<const SiStripRecHit2D* >(&(*theHits[0]));

    std::array<float,20> pixelADC_,pixelADCx_,pixelADCy_;

    int n = 0;
    for(auto& hit: theHits)
    {
       n++;
       TrackerSingleRecHit* hit = dynamic_cast<TrackerSingleRecHit* >((*recHit));

       if(!hit) continue;

       const GeomDet* gDet = (hit)->det();
       float x = (hit)->globalState().position.x();
       float y = (hit)->globalState().position.y();
       float z = (hit)->globalState().position.z();
       float phi = (hit)->globalState().position.phi;
       float r = (hit)->globalState().position.r;

       ax1 = gDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
       ax2 = gDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();

       SiPixelRecHit *pixHit = dynamic_cast<SiPixelRecHit*>(hit);

       if(pixHit)
       {

          std::array<float,13> pixInf;
          auto clust = pixHit->cluster();

          //Tp Matching
          auto tpRange = tpClust->equal_range(hits[0]->firstClusterRef());

          // std::cout << "Doublet no. "  << i << " hit no. " << lIt->doublets().innerHitId(i) << std::endl;

          std::array< std::pair<int,int>,3> tParticle;
          tParticle[0] = {-1.0,-1.0}; tParticle[1] = {-1.0,-1.0}; tParticle[2] = {-1.0,-1.0};

          int numTP = tpRange.second - tpRange.first;
          numTP = -std::max(numTP,3);

          int tpCounter = 0;
          for(auto ip=tpRange.first; ip != tpRange.second; ++ip)
          {
            tParticle[tpCounter] = {ip->second.key(),(*ip->second).pdgId()};
          }


          pixInf[0] = (float) n;


          pixInf[1] = (float)clust->x();
          pixInf[2] = (float)clust->y();
          pixInf[3] = (float)clust->size();
          pixInf[4] = (float)clust->sizeX();
          pixInf[5] = (float)clust->sizeY();
          pixInf[6] = (float)clust->charge();

          pixInf[7] = (float)clust->sizeX() > 10.;
          pixInf[8] = (float)clust->sizeY() > 10.;
          pixInf[9] = (float)clust->sizeY() / (float)clust->sizeX();

          pixInf[10]  = (float)pixHit->spansTwoROCs();
          pixInf[11] = (float)pixHit->hasBadPixels();
          pixInf[12] = (float)pixHit->isOnEdge();

          int minSize = -std::max(-clust->size(),-20);
          for (int k = 0; k<20; ++k)
          {
            pixelADC_[k]  = -1.0;
            pixelADCx_[k] = -1.0;
            pixelADCy_[k] = -1.0;
          }
          for (int k = 0; k < minSize; ++k)
          {
            pixelADC_[k]  = (float)clust->pixel(k).adc;
            pixelADCx_[k] = (float)clust->pixel(k).x;
            pixelADCy_[k] = (float)clust->pixel(k).y;

          }

          outHitFile << eveNumber << "\t" << runNumber << "\t" << lumNumber << "\t";
          outHitFile << layer.subdet << "\t" << float(layer.side) << "\t" << float(layer.idLayer);

          for(auto& p: pixInf)
            outHitFile << "\t" << p;
          for(auto& p: pixelADC_)
          {
            std::cout << p;
            outHitFile << "\t" << p;
          }
          std::cout<<std::endl;

          for(auto& p: pixelADCx_)
            outHitFile << "\t" << p;
          for(auto& p: pixelADCy_)
            outHitFile << "\t" << p;


          //std::cout << " Si Pixel Rec Hit" << std::endl;

          n++;
          continue;
        }

        // const SiStripRecHit2D* siStripHit2D = dynamic_cast<SiStripRecHit2D const *>(hit);
        // if(siStripHit2D)
        // {
        //    //dimension first ampsize charge merged splitClusterError id
        //    std::array<float,7> stripInf;
        //
        //    auto clust = siStripHit2D->cluster();
        //
        //    stripInf[0] = float(n);
        //    stripInf[1] = 2.0;
        //
        //    stripInf[2] = clust->charge();
        //    stripInf[3] = clust->barycenter();
        //    stripInf[4] = clust->firstStrip();
        //    stripInf[5] = clust->isMerged();
        //    stripInf[6] = clust->amplitudes().size();
        //    stripInfos_.push_back(stripInf);
        //
        //    int minSize = -std::max(-clust->amplitudes().size(),-20);
        //
        //    for (int k = 0; k<20; ++k)
        //      stripADC_[k]  = -1.0;
        //    for (int k = 0; k < minSize; ++k)
        //      stripADC_[k]  = (float)clust->amplitudes()[k];
        //    continue;
        // }
        // const SiStripRecHit1D* siStripHit1D = dynamic_cast<SiStripRecHit1D const *>(hit);
        // if(siStripHit1D)
        //  {
        //    //dimension first ampsize charge merged splitClusterError id
        //    std::array<float,7> stripInf;
        //
        //    auto clust = siStripHit1D->cluster();
        //
        //    stripInf[0] = float(n);
        //    stripInf[1] = 1.0;
        //
        //    stripInf[2] = clust->charge();
        //    stripInf[3] = clust->barycenter();
        //    stripInf[4] = clust->firstStrip();
        //    stripInf[5] = clust->isMerged();
        //    stripInf[6] = clust->amplitudes().size();
        //    stripInfos_.push_back(stripInf);
        //
        //    int minSize = -std::max(-clust->amplitudes().size(),-20);
        //
        //    for (int k = 0; k<20; ++k)
        //      stripADC_[k]  = -1.0;
        //    for (int k = 0; k < minSize; ++k)
        //      stripADC_[k]  = (float)clust->amplitudes()[k];
        //    continue;
        //  }

    }

    ret->addHits(layer.nameIndex, layer.extractor->hits((const TkTransientTrackingRecHitBuilder &)(*theTTRHBuilders[layer.nameIndex]), ev, es));
  }
  ret->shrink_to_fit();

  return ret;
}
//new function for FastSim only
std::unique_ptr<SeedingLayerSetsHits> SeedingLayerSetsBuilder::makeSeedingLayerSetsHitsforFastSim(const edm::Event& ev, const edm::EventSetup& es) {
  updateEventSetup(es);

  edm::Handle<FastTrackerRecHitCollection> fastSimrechits_;
  ev.getByToken(fastSimrecHitsToken_,fastSimrechits_); //using FastSim RecHits
  edm::ESHandle<TrackerTopology> trackerTopology;
  es.get<TrackerTopologyRcd>().get(trackerTopology);
  const TrackerTopology* const tTopo = trackerTopology.product();
  SeedingLayerSetsHits::OwnedHits layerhits_;

  auto ret = std::make_unique<SeedingLayerSetsHits>(theNumberOfLayersInSet,
                                                    &theLayerSetIndices,
                                                    &theLayerNames,
                                                    &theLayerDets);

  for(auto& layer: theLayers) {
    layerhits_.clear();
    for(auto &rh : *fastSimrechits_){
      GeomDetEnumerators::SubDetector subdet = GeomDetEnumerators::invalidDet;
      TrackerDetSide side = TrackerDetSide::Barrel;
      int idLayer = 0;
      if( (rh.det()->geographicalId()).subdetId() == PixelSubdetector::PixelBarrel){
      	subdet = GeomDetEnumerators::PixelBarrel;
	side = TrackerDetSide::Barrel;
	idLayer = tTopo->pxbLayer(rh.det()->geographicalId());
      }
      else if ((rh.det()->geographicalId()).subdetId() == PixelSubdetector::PixelEndcap){
   	subdet = GeomDetEnumerators::PixelEndcap;
	idLayer = tTopo->pxfDisk(rh.det()->geographicalId());
	if(tTopo->pxfSide(rh.det()->geographicalId())==1)
	  side = TrackerDetSide::NegEndcap;
	else
	  side = TrackerDetSide::PosEndcap;
      }

      if(layer.subdet == subdet && layer.side == side && layer.idLayer == idLayer){
	BaseTrackerRecHit const & b(rh);
	auto ptrHit = (BaseTrackerRecHit *)(b.clone());
	layerhits_.emplace_back(ptrHit);
      }
      else continue;
    }
    ret->addHits(layer.nameIndex, std::move(layerhits_));
  }
  ret->shrink_to_fit();
  return ret;
}
