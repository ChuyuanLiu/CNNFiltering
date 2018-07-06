// -*- C++ -*-
//
// Package:    CNNFiltering/CNNAnalyze
// Class:      CNNAnalyze
//
/**\class CNNAnalyze CNNAnalyze.cc CNNFiltering/CNNAnalyze/plugins/CNNAnalyze.cc

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
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"

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

class CNNAnalyze : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit CNNAnalyze(const edm::ParameterSet&);
  ~CNNAnalyze();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  int particleBit();

  // ----------member data ---------------------------

  int doubletSize;
  std::string processName_;
  edm::EDGetTokenT<edm::View<reco::Track>> alltracks_;
  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
  edm::EDGetTokenT<reco::BeamSpot>  bsSrc_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  // edm::GetterOfProducts<IntermediateHitDoublets> getterOfProducts_;

  float padHalfSize;
  int padSize, tParams;

  TTree* cnntree;

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
CNNAnalyze::CNNAnalyze(const edm::ParameterSet& iConfig):
processName_(iConfig.getParameter<std::string>("processName")),
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap")))
{

  usesResource("TFileService");

  edm::Service<TFileService> fs;
  cnntree = fs->make<TTree>("CNNTree","Doublets Tree");

  cnntree->Branch("test",      &test,          "test/I");

  edm::InputTag beamSpotTag = iConfig.getParameter<edm::InputTag>("beamSpot");
  bsSrc_ = consumes<reco::BeamSpot>(beamSpotTag);

  infoPileUp_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter< edm::InputTag >("infoPileUp"));

  padHalfSize = 8;
  padSize = (int)(padHalfSize*2);
  tParams = 26;

}


CNNAnalyze::~CNNAnalyze()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
CNNAnalyze::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
  // std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

  // std::cout<<"CNNDoublets Analyzer"<<std::endl;

  edm::Handle<View<reco::Track> >  trackCollection;
  iEvent.getByToken(alltracks_, trackCollection);

  edm::Handle<ClusterTPAssociation> tpClust;
  iEvent.getByToken(tpMap_,tpClust);

  // test = iEvent.id().event();
  //
  // cnntree->Fill();

  int eveNumber = iEvent.id().event();
  int runNumber = iEvent.id().run();
  int lumNumber = iEvent.id().luminosityBlock();

  std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31}; //seqNumbers of pixel detectors 0,1,2,3 barrel 14,15,16, fwd 29,30,31 bkw
  std::vector<int> partiList{11,13,15,22,111,211,311,321,2212,2112,3122,223};

  // reco::Vertex thePrimaryV, theBeamSpotV;

  //The Beamspot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(bsSrc_,recoBeamSpotHandle);
  reco::BeamSpot const & bs = *recoBeamSpotHandle;
  // reco::Vertex theBeamSpotV(bs.position(), bs.covariance3D());

  edm::Handle< std::vector<PileupSummaryInfo> > puinfoH;
  iEvent.getByToken(infoPileUp_,puinfoH);
  PileupSummaryInfo puinfo;

  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }

  int puNumInt = puinfo.getPU_NumInteractions();

  for(edm::View<reco::Track>::size_type i=0; i<trackCollection->size(); ++i)
  {
    std::map<int,const TrackerSingleRecHit*> theHits;
    std::map<int,bool> flagHit,isBad,isEdge,isBig;
    std::map<int,int> hitSize;

    auto track = trackCollection->refAt(i);
    auto hitPattern = track->hitPattern();
    int pixHits = hitPattern.numberOfValidPixelHits();
    test = pixHits;

    if(pixHits < 4)
      continue;

    float pt    = track->pt();
    float eta   = track->eta();
    float phi   = track->phi();
    float p     = track->p();
    float chi2n = track->normalizedChi2();
    int   nhit  = track->numberOfValidHits();
    float d0    = track->d0();
    float dz    = track->dz();

    int nhpxb   = hitPattern.numberOfValidPixelBarrelHits();
    int nhpxf   = hitPattern.numberOfValidPixelEndcapHits();
    int nhtib   = hitPattern.numberOfValidStripTIBHits();
    int nhtob   = hitPattern.numberOfValidStripTOBHits();
    int nhtid   = hitPattern.numberOfValidStripTIDHits();
    int nhtec   = hitPattern.numberOfValidStripTECHits();

    for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
    {
      TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(*recHit);

      DetId detId = (*recHit)->geographicalId();
      unsigned int subdetid = detId.subdetId();

      if(detId.det() != DetId::Tracker) continue;
      if (!((subdetid==1) || (subdetid==2))) continue;

      const SiPixelRecHit* pixHit = dynamic_cast<SiPixelRecHit const *>(hit);

      int hitLayer = -1;

      if(subdetid==1) //barrel
        hitLayer = PXBDetId(detId).layer();
      else
      {
        int side = PXFDetId(detId).side();
        float z = (hit->globalState()).position.z();

        if(fabs(z)>28.0) hitLayer = 4;
        if(fabs(z)>36.0) hitLayer = 5;
        if(fabs(z)>44.0) hitLayer = 6;

        if(side==2.0) hitLayer +=3;
      }

      if (pixHit && hitLayer >= 0)
      {
        bool thisBad,thisEdge,thisBig;

        auto thisClust = pixHit->cluster();
        int thisSize   = thisClust->size();

        thisBig  = pixHit->spansTwoROCs();
        thisBad  = pixHit->hasBadPixels();
        thisEdge = pixHit->isOnEdge();

        bool keepThis = false;

        if(flagHit.find(hitLayer) != flagHit.end())
        {
          //Keeping the good,not on edge,not big, with higher charge
          if(isBad[hitLayer] || isEdge[hitLayer] || isBig[hitLayer])
            if(!(thisBad || thisEdge || thisBig))
              keepThis = true;
          if(isBad[hitLayer] && !thisBad)
              keepThis = true;
          if((isBad[hitLayer] && thisBad)||(!(isBad[hitLayer] || thisBad)))
            if(thisSize > hitSize[hitLayer])
              keepThis = true;
          if(thisBad && !isBad[hitLayer])
              keepThis = false;
        }else
          keepThis = true;

        if(keepThis)
          theHits[hitLayer] = hit;

        flagHit[hitLayer] = true;

      }


    }

    for (auto& h: theHits)
    {
        std::cout << h.first << ": " << h.second->geographicalId().subdetId() << '\n';
    }

  }



}


// ------------ method called once each job just before starting event loop  ------------
void
CNNAnalyze::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
CNNAnalyze::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNNAnalyze::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNAnalyze);
