// -*- C++ -*-
//
// Package:    CNNFiltering/CNNParticleId
// Class:      CNNParticleId
//
/**\class CNNParticleId CNNParticleId.cc CNNFiltering/CNNParticleId/plugins/CNNParticleId.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  adrianodif
//         Created:  Tue, 30 Jan 2018 12:05:21 GMT
//
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

// system include files
#include <memory>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include <iostream>
#include <string>
#include <fstream>

#include "TH2F.h"
#include "TTree.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"

#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "CommonTools/Utils/interface/associationMapFilterValues.h"
#include "FWCore/Utilities/interface/IndexSet.h"
#include <type_traits>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class CNNParticleId : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit CNNParticleId(const edm::ParameterSet&);
  ~CNNParticleId();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  float hashId(float kId, float pId, float mId, float eId, float elseId);
  int particleBit();

  // ----------member data ---------------------------

  int doubletSize;
  std::string processName_;
  edm::EDGetTokenT<edm::View<reco::Track>> alltracks_;
  int minPix_;

  edm::EDGetTokenT<reco::BeamSpot>  bsSrc_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  // edm::GetterOfProducts<IntermediateHitDoublets> getterOfProducts_;

  float padHalfSize;
  int padSize, tParams;

  float pt, eta, phi, p, chi2n, d0, dx, dz, sharedFraction,sharedMomFraction;
  int nhit, nhpxf, nhtib, nhtob, nhtid, nhtec, nhpxb, nHits, trackPdg,trackMomPdg;
  int eveNumber, runNumber, lumNumber;

  std::vector<float>  x, y, z, phi_hit, r, c_x, c_y, charge, ovfx, ovfy;
  std::vector<float> ratio, pdgId, motherPdgId, size, sizex, sizey;
  //std::vector<TH2> hitClust;

  std::vector<float> hitPixel0, hitPixel1, hitPixel2, hitPixel3, hitPixel4;
  std::vector<float> hitPixel5, hitPixel6, hitPixel7, hitPixel8, hitPixel9;

  std::vector< std::vector<float> > hitPixels;

  // TTree* cnntree;

  UInt_t test;


};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CNNParticleId::CNNParticleId(const edm::ParameterSet& iConfig):
processName_(iConfig.getParameter<std::string>("processName")),
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
minPix_(iConfig.getParameter<int>("minPix"))
{

  padHalfSize = 7.5;
  padSize = (int)(padHalfSize*2);
  tParams = 26;

  hitPixels.push_back(hitPixel0);
  hitPixels.push_back(hitPixel1);
  hitPixels.push_back(hitPixel2);
  hitPixels.push_back(hitPixel3);
  hitPixels.push_back(hitPixel4);
  hitPixels.push_back(hitPixel5);
  hitPixels.push_back(hitPixel6);
  hitPixels.push_back(hitPixel7);
  hitPixels.push_back(hitPixel8);
  hitPixels.push_back(hitPixel9);

  for(int i = 0; i<10;i++)
    for(int j =0;j<padSize*padSize;j++)
      hitPixels[i].push_back(0.0);

  for(int i = 0; i<10;i++)
  {
    x.push_back(0.0);
    y.push_back(0.0);
    z.push_back(0.0);
    phi_hit.push_back(0.0);
    r.push_back(0.0);
    c_x.push_back(0.0);
    c_y.push_back(0.0);
    pdgId.push_back(0.0);
    motherPdgId.push_back(0.0);
    size.push_back(0);
    sizex.push_back(0);
    sizey.push_back(0);
    charge.push_back(0);
    ovfx.push_back(0.0);
    ovfy.push_back(0.0);
    ratio.push_back(0.0);

  }




}


CNNParticleId::~CNNParticleId()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

float CNNParticleId::hashId(float kId, float pId, float mId, float eId, float elseId)
{

  int p1 = int(kId*1E3);
  int p2 = int(pId*1E6);
  int p3 = int(mId*1E9);
  int p4 = int(eId*1E12);
  int p5 = int(elseId*1E15);

  return float(p1+p2+p3+p4+p5)/1E16;

}
// ------------ method called for each event  ------------
void
CNNParticleId::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
  // std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

  // std::cout<<"CNNDoublets Analyzer"<<std::endl;

  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality= reco::TrackBase::qualityByName(theTrackQuality);

  edm::Handle<View<reco::Track> >  trackCollection;
  iEvent.getByToken(alltracks_, trackCollection);

  eveNumber = iEvent.id().event();
  runNumber = iEvent.id().run();
  lumNumber = iEvent.id().luminosityBlock();

  std::string fileName = processName_ + ".txt";
  //std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
  //fileName += "_" + processName_ + "_dnn_doublets.txt";
  std::ofstream outCNNFile(fileName, std::ofstream::app);

  std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31}; //seqNumbers of pixel detectors 0,1,2,3 barrel 14,15,16, fwd 29,30,31 bkw
  std::vector<int> partiList{11,13,15,22,111,211,311,321,2212,2112,3122,223};


  for(edm::View<reco::Track>::size_type tt=0; tt<trackCollection->size(); ++tt)
  {
    std::vector<double> theData;
    // std::cout << "Track ------------------- "<< std::endl;
    // std::cout << std::endl;
    std::map<int,const TrackerSingleRecHit*> theHits;
    std::map<int,bool> isBad,isEdge,isBig;
    std::map<int,double> hitSize,pdgIds,flagHit;
    std::map<double,int> pdgMap,pdgMomMap;

    auto track = trackCollection->refAt(tt);
    auto hitPattern = track->hitPattern();
    bool trkQual  = track->quality(trackQuality);
    
    float pId = hashId(0.1,0.2,0.3,0.4,0.5);
    track->setParticleId(pId);std::cout << ".";
    // track->setPionId(0.2);
    // track->setMuonId(0.3);
    // track->setElecId(0.4);
    // track->setElseId(0.5);

    sharedFraction = 0.0;
    nHits = 0;

    for(int i = 0; i<10;i++)
      for(int j =0;j<padSize*padSize;j++)
        hitPixels[i][j] = 0.0;

    for(int i = 0; i<10;i++)
    {
      x[i] = 0.0;
      y[i] = 0.0;
      z[i] = 0.0;
      phi_hit[i] = 0.0;
      r[i] = 0.0;
      c_x[i] = 0.0;
      c_y[i] = 0.0;
      pdgId[i] = 0.0;
      motherPdgId[i] = 0.0;
      size[i] = 0.0;
      sizex[i] = 0.0;
      sizey[i] = 0.0;
      charge[i] = 0.0;
      ovfx[i] = 0.0;
      ovfy[i] = 0.0;
      ratio[i] = 0.0;

    }
    // bool isSimMatched = false;
    //
    // auto tpFound = recSimColl.find(track);
    // isSimMatched = tpFound != recSimColl.end();
    //
    // if (isSimMatched) {
    //     const auto& tp = tpFound->val;
    //     //nSimHits = tp[0].first->numberOfTrackerHits();
    //     sharedFraction = tp[0].second;
    // }
    // if(isSimMatched)
    //   std::cout<< "Good Track - "<<sharedFraction<<std::endl;
    // else
    //   std::cout<< "Bad Track"<<std::endl;

    if(!trkQual)
      continue;
    // std::cout << "- Track Quality " <<std::endl;
    int pixHits = hitPattern.numberOfValidPixelHits();
    // std::cout << "- No Pixel Hits :" << pixHits << std::endl;
    if(pixHits < minPix_)
    {
      track->setKaonId(0.0);
      track->setPionId(0.0);
      // track->setMuonId(0.0);
      // track->setElecId(0.0);
      // track->setElseId(0.0);
      continue;
    }

    track->setKaonId(0.1);
    track->setPionId(0.1);


  }

// std::cout << "Closing" << std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void
CNNParticleId::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
CNNParticleId::endJob()
{
  // std::cout << "Closing" << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNNParticleId::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNParticleId);
