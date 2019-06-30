// -*- C++ -*-
//
// Package:    CNNFiltering/CNNTrackAnalyze
// Class:      CNNTrackAnalyze
//
/**\class CNNTrackAnalyze CNNTrackAnalyze.cc CNNFiltering/CNNTrackAnalyze/plugins/CNNTrackAnalyze.cc

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


std::array< std::array < std::array <float,2>, 2>, 73>allTheBins = {
{{-28.0,28.0},{2.5,5.0}},
{{-28.0,28.0},{6.0,8.0}},
{{-28.0,28.0},{10.0,12.5}},
{{-28.0,28.0},{15.0,17.5}},
{{-35.0,-29.0},{2.5,17.5}},
{{29.0,35.0},{2.5,17.5}},
{{-44.0,-36.0},{2.5,17.5}},
{{36.0,44.0},{2.5,17.5}},
{{-53.0,-44.5},{2.5,17.5}},
{{44.5,53.0},{2.5,17.5}},
{{-65.0,65.0},{23.0,25.5}},
{{-65.0,65.0},{25.5,30.0}},
{{-65.0,65.0},{30.5,33.5}},
{{-65.0,65.0},{33.5,38.0}},
{{-65.0,65.0},{38.1,42.0}},
{{-65.0,65.0},{42.0,46.0}},
{{-65.0,65.0},{46.5,50.0}},
{{-65.0,65.0},{50.1,53.0}},
{{-65.0,65.0},{57.1,61.0}},
{{-65.0,65.0},{61.1,65.0}},
{{-65.0,65.0},{65.5,69.0}},
{{-65.0,65.0},{69.1,73.0}},
{{-65.0,65.0},{75.0,78.0}},
{{-65.0,65.0},{78.5,83.5}},
{{-65.0,65.0},{83.6,87.0}},
{{-65.0,65.0},{87.1,90.0}},
{{-65.0,65.0},{92.0,96.5}},
{{-115.0,-70.0},{25.5,30.0}},
{{70.0,115.0},{25.5,30.0}},
{{-115.0,-70.0},{33.1,38.0}},
{{70.0,115.0},{33.1,38.0}},
{{-65.0,65.0},{96.8,103.0}},
{{-115.0,-70.0},{42.0,46.0}},
{{70.0,115.0},{42.0,46.0}},
{{-115.0,-70.0},{57.1,61.0}},
{{70.0,115.0},{57.1,61.0}},
{{-65.0,65.0},{105.0,115.0}},
{{-115.0,-70.0},{61.1,65.0}},
{{70.0,115.0},{61.1,65.0}},
{{-115.0,-70.0},{65.5,69.0}},
{{70.0,115.0},{65.5,69.0}},
{{-115.0,-70.0},{69.1,73.0}},
{{70.0,115.0},{69.1,73.0}},
{{-115.0,-70.0},{75.0,78.0}},
{{70.0,115.0},{75.0,78.0}},
{{-115.0,-70.0},{78.5,83.5}},
{{70.0,115.0},{78.5,83.5}},
{{-115.0,-70.0},{83.6,87.0}},
{{70.0,115.0},{83.6,87.0}},
{{-115.0,-70.0},{87.1,90.0}},
{{70.0,115.0},{87.1,90.0}},
{{-115.0,-70.0},{92.0,96.5}},
{{70.0,115.0},{92.0,96.5}},
{{-115.0,-70.0},{96.8,103.0}},
{{70.0,115.0},{96.8,103.0}},
{{-115.0,-70.0},{105.0,115.0}},
{{70.0,115.0},{105.0,115.0}},
{{-170.0,-120.0},{25.5,30.0}},
{{120.0,170.0},{25.5,30.0}},
{{-230.0,-120.0},{33.1,38.0}},
{{120.0,230.0},{33.1,38.0}},
{{120.0,170.0},{105.0,115.0}},
{{-170.0,-120.0},{105.0,115.0}},
{{-280.0,-120.0},{42.0,46.0}},
{{120.0,280.0},{42.0,46.0}},
{{-280.0,-120.0},{54.5,57.0}},
{{120.0,280.0},{54.5,57.0}},
{{-280.0,-120.0},{65.5,69.0}},
{{120.0,280.0},{65.5,69.0}},
{{-280.0,-120.0},{77.1,83.5}},
{{120.0,280.0},{77.1,83.5}},
{{-280.0,-120.0},{96.8,103.0}},
{{120.0,280.0},{96.8,103.0}}
};

class CNNTrackAnalyze : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit CNNTrackAnalyze(const edm::ParameterSet&);
  ~CNNTrackAnalyze();

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
  edm::EDGetTokenT<reco::GenParticleCollection> genParticles_;
  edm::EDGetTokenT<TrackingParticleCollection> traParticles_;
  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trMap_;
  edm::EDGetTokenT<reco::TrackToGenParticleAssociator> genMap_;
  edm::EDGetTokenT<reco::BeamSpot>  bsSrc_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  // edm::GetterOfProducts<IntermediateHitDoublets> getterOfProducts_;

  float padHalfSize;
  int padSize, tParams;

  float pt, eta, phi, p, chi2n, d0, dx, dz, sharedFraction,sharedMomFraction;
  int nhit, nhpxf, nhtib, nhtob, nhtid, nhtec, nhpxb, nHits, trackPdg,trackMomPdg;
  int eveNumber, runNumber, lumNumber;

  float dummy;

  //General Hit
  std::array<float,73> x, y, z, phi_hit, r, charge, n_seq;
  std::array<float,73> ax1,ax2,ax3,ax4,dZ,detId;
  std::array<float,73> pdgId, motherPdgId,rawId;

  //Pixel Hit
  std::array<float,10> p_size, p_sizex, p_sizey, p_x, p_y, p_ovx, p_ovy;
  std::array<float,10> p_skew, p_big, p_bad, p_edge, p_charge;

  // std::array<float,256> hitPixel0, hitPixel1, hitPixel2, hitPixel3, hitPixel4;
  // std::array<float,256> hitPixel5, hitPixel6, hitPixel7, hitPixel8, hitPixel9;

  std::array< std::array<float,256>, 10> hitPixels;

  //Strip Hits
  std::array<float,63> s_dim, s_center, s_first, s_merged, s_size, s_charge;
  // std::array<float,16> hitStrip0, hitStrip1, hitStrip2, hitStrip3, hitStrip4;
  // std::array<float,16> hitStrip5, hitStrip6, hitStrip7, hitStrip8, hitStrip9, hitStrip10, hitStrip11;
  // std::array<float,16> hitStrip12, hitStrip13, hitStrip14, hitStrip15, hitStrip16, hitStrip17, hitStrip18;
  // std::array<float,16> hitStrip19, hitStrip20, hitStrip21, hitStrip22, hitStrip23, hitStrip24, hitStrip25;
  // std::array<float,16> hitStrip26, hitStrip27, hitStrip28, hitStrip29, hitStrip30, hitStrip31, hitStrip32;
  // std::array<float,16> hitStrip33, hitStrip34, hitStrip35, hitStrip36, hitStrip37, hitStrip38, hitStrip39;
  // std::array<float,16> hitStrip40, hitStrip41, hitStrip42, hitStrip43, hitStrip44, hitStrip45, hitStrip46;
  // std::array<float,16> hitStrip47, hitStrip48, hitStrip49, hitStrip50, hitStrip51, hitStrip52, hitStrip53;
  // std::array<float,16> hitStrip54, hitStrip55, hitStrip56, hitStrip57, hitStrip58, hitStrip59, hitStrip60;
  // std::array<float,16> hitStrip61, hitStrip62;

  std::array< std::array<float, 16>, 62 > hitStrips;

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
CNNTrackAnalyze::CNNTrackAnalyze(const edm::ParameterSet& iConfig):
processName_(iConfig.getParameter<std::string>("processName")),
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
genParticles_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
traParticles_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("traParticles"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap"))),
trMap_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getParameter<edm::InputTag>("trMap"))),
genMap_(consumes<reco::TrackToGenParticleAssociator>(iConfig.getParameter<edm::InputTag>("genMap")))
{

  dummy = -0.000053421269;
  padHalfSize = 7.5;
  padSize = (int)(padHalfSize*2);
  tParams = 26;

  // hitPixels.push_back(hitPixel0);
  // hitPixels.push_back(hitPixel1);
  // hitPixels.push_back(hitPixel2);
  // hitPixels.push_back(hitPixel3);
  // hitPixels.push_back(hitPixel4);
  // hitPixels.push_back(hitPixel5);
  // hitPixels.push_back(hitPixel6);
  // hitPixels.push_back(hitPixel7);
  // hitPixels.push_back(hitPixel8);
  // hitPixels.push_back(hitPixel9);

  for(int i = 0; i<10;i++)
    for(int j =0;j<256;j++)
      hitPixels[i][j] = dummy;

  for(int i = 0; i<63;i++)
    for(int j =0;j<16;j++)
      hitPixels[i][j] = dummy;

  // hitStrips.push_back(hitStrip0);
  // hitStrips.push_back(hitStrip1);
  // hitStrips.push_back(hitStrip2);
  // hitStrips.push_back(hitStrip3);
  // hitStrips.push_back(hitStrip4);
  // hitStrips.push_back(hitStrip5);
  // hitStrips.push_back(hitStrip6);
  // hitStrips.push_back(hitStrip7);
  // hitStrips.push_back(hitStrip8);
  // hitStrips.push_back(hitStrip9);
  // hitStrips.push_back(hitStrip10);
  // hitStrips.push_back(hitStrip11);
  // hitStrips.push_back(hitStrip12);
  // hitStrips.push_back(hitStrip13);
  // hitStrips.push_back(hitStrip14);
  // hitStrips.push_back(hitStrip15);
  // hitStrips.push_back(hitStrip16);
  // hitStrips.push_back(hitStrip17);
  // hitStrips.push_back(hitStrip18);
  // hitStrips.push_back(hitStrip19);
  // hitStrips.push_back(hitStrip20);
  // hitStrips.push_back(hitStrip21);
  // hitStrips.push_back(hitStrip22);
  // hitStrips.push_back(hitStrip23);
  // hitStrips.push_back(hitStrip24);
  // hitStrips.push_back(hitStrip25);
  // hitStrips.push_back(hitStrip26);
  // hitStrips.push_back(hitStrip27);
  // hitStrips.push_back(hitStrip28);
  // hitStrips.push_back(hitStrip29);
  // hitStrips.push_back(hitStrip30);
  // hitStrips.push_back(hitStrip31);
  // hitStrips.push_back(hitStrip32);
  // hitStrips.push_back(hitStrip33);
  // hitStrips.push_back(hitStrip34);
  // hitStrips.push_back(hitStrip35);
  // hitStrips.push_back(hitStrip36);
  // hitStrips.push_back(hitStrip37);
  // hitStrips.push_back(hitStrip38);
  // hitStrips.push_back(hitStrip39);
  // hitStrips.push_back(hitStrip40);
  // hitStrips.push_back(hitStrip41);
  // hitStrips.push_back(hitStrip42);
  // hitStrips.push_back(hitStrip43);
  // hitStrips.push_back(hitStrip44);
  // hitStrips.push_back(hitStrip45);
  // hitStrips.push_back(hitStrip46);
  // hitStrips.push_back(hitStrip47);
  // hitStrips.push_back(hitStrip48);
  // hitStrips.push_back(hitStrip49);
  // hitStrips.push_back(hitStrip50);
  // hitStrips.push_back(hitStrip51);
  // hitStrips.push_back(hitStrip52);
  // hitStrips.push_back(hitStrip53);
  // hitStrips.push_back(hitStrip54);
  // hitStrips.push_back(hitStrip55);
  // hitStrips.push_back(hitStrip56);
  // hitStrips.push_back(hitStrip57);
  // hitStrips.push_back(hitStrip58);
  // hitStrips.push_back(hitStrip59);
  // hitStrips.push_back(hitStrip60);
  // hitStrips.push_back(hitStrip61);
  // hitStrips.push_back(hitStrip62);

  // for(int i = 0; i<30;i++)
  // {
  //   x.push_back(0.0);
  //   y.push_back(0.0);
  //   z.push_back(0.0);
  //   phi_hit.push_back(0.0);
  //   r.push_back(0.0);
  //   c_x.push_back(0.0);
  //   c_y.push_back(0.0);
  //   pdgId.push_back(0.0);
  //   motherPdgId.push_back(0.0);
  //   size.push_back(0);
  //   sizex.push_back(0);
  //   sizey.push_back(0);
  //   charge.push_back(0);
  //   ovfx.push_back(0.0);
  //   ovfy.push_back(0.0);
  //   ratio.push_back(0.0);
  //
  // }


  edm::InputTag beamSpotTag = iConfig.getParameter<edm::InputTag>("beamSpot");
  bsSrc_ = consumes<reco::BeamSpot>(beamSpotTag);

  infoPileUp_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter< edm::InputTag >("infoPileUp"));



}


CNNTrackAnalyze::~CNNTrackAnalyze()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
CNNTrackAnalyze::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
  // std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

  // std::cout<<"CNNDoublets Analyzer"<<std::endl;

  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality= reco::TrackBase::qualityByName(theTrackQuality);

  edm::Handle<View<reco::Track> >  trackCollection;
  iEvent.getByToken(alltracks_, trackCollection);

  edm::Handle<ClusterTPAssociation> tpClust;
  iEvent.getByToken(tpMap_,tpClust);

  edm::Handle<reco::TrackToTrackingParticleAssociator> tpTracks;
  iEvent.getByToken(trMap_, tpTracks);


  eveNumber = iEvent.id().event();
  runNumber = iEvent.id().run();
  lumNumber = iEvent.id().luminosityBlock();

  std::string fileName = std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber) + processName_ + ".txt";
  //fileName += "_" + processName_ + "_dnn_doublets.txt";
  std::ofstream trackFile(fileName, std::ofstream::app);

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

    sharedFraction = 0.0;
    nHits = 0;

    for(int i = 0; i<10;i++)
      for(int j =0;j<padSize*padSize;j++)
        hitPixels[i][j] = 0.0;

    for(int i = 0; i<73;i++)
    {
      x[i] = dummy;
      y[i] = dummy;
      z[i] = dummy;
      phi_hit[i] = dummy;
      r[i] = dummy;
      charge[i] = dummy;
      n_seq[i] = dummy;
      pdgId[i] = dummy;
      motherPdgId[i] = dummy;
      dZ[i] = dummy;
      ax1[i] = dummy;
      ax2[i] = dummy;
      ax3[i] = dummy;
      ax4[i] = dummy;
      rawId[i] = dummy;
    }

    for(int i = 0; i<10;i++)
    {
      p_size[i]  = dummy;
      p_sizex[i]  = dummy;
      p_sizey[i]  = dummy;
      p_x[i]  = dummy;
      p_y[i]  = dummy;
      p_ovx[i]  = dummy;
      p_ovy[i]  = dummy;
      p_skew[i]  = dummy;
      p_big[i]  = dummy;
      p_bad[i]  = dummy;
      p_edge[i]  = dummy;
      p_charge[i] = dummy;
    }

    for(int i = 0; i<63;i++)
    {
      s_dim[i] = dummy;
      s_center[i] = dummy;
      s_first[i] = dummy;
      s_merged[i] = dummy;
      s_size[i] = dummy;
      s_charge[i] = dummy;
    }

    for(int i = 0; i<10;i++)
      for(int j =0;j<256;j++)
        hitPixels[i][j] = dummy;

    for(int i = 0; i<63;i++)
      for(int j =0;j<16;j++)
        hitStrips[i][j] = dummy;


    if(!trkQual)
      continue;
    // std::cout << "- Track Quality " <<std::endl;
    int pixHits = hitPattern.numberOfValidPixelHits();
    // std::cout << "- No Pixel Hits :" << pixHits << std::endl;
    if(pixHits < 2)
      continue;

    theData.push_back((double)eveNumber);
    theData.push_back((double)runNumber);
    theData.push_back((double)lumNumber);

    theData.push_back((double)track->pt());
    theData.push_back((double)track->eta());
    theData.push_back((double)track->phi());
    theData.push_back((double)track->p());
    theData.push_back((double)track->normalizedChi2());
    theData.push_back((double)track->numberOfValidHits());
    theData.push_back((double)track->d0());
    theData.push_back((double)track->dz());

    theData.push_back((double)hitPattern.numberOfValidPixelBarrelHits());
    theData.push_back((double)hitPattern.numberOfValidPixelEndcapHits());
    theData.push_back((double)hitPattern.numberOfValidStripTIBHits());
    theData.push_back((double)hitPattern.numberOfValidStripTOBHits());
    theData.push_back((double)hitPattern.numberOfValidStripTIDHits());
    theData.push_back((double)hitPattern.numberOfValidStripTECHits());

    int hitCounter = -1;
    for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
    {

      TrackerSingleRecHit const * h= dynamic_cast<TrackerSingleRecHit const *>(*recHit);

      if(!h)
        continue;

      DetId detId = (*recHit)->geographicalId();
      unsigned int subdetid = detId.subdetId();
      if(detId.det() != DetId::Tracker) continue;
      ++hitCounter;

      int hitBin = -1;
      float hit_r = (h->globalState()).position.r();
      float hit_z = (h->globalState()).position.z();

      for (int i =0; i<73;i++)
      {
        if(r<=allTheBins[1][1] && r>=allTheBins[1][0])
        {
          if(r<=allTheBins[1][1] && r>=allTheBins[1][0])
          {
              hitBin = 1
              break;
          }
        }
      }

      if (hitBin < 0 || hitBin>72) continue;

      const GeomDet* gDet = (hit)->det();


      //Pixel Hit!
      if(hitBin < 10)
      {

        const SiPixelRecHit* hh = dynamic_cast<SiPixelRecHit const *>(hit);
        if(!pixHit) continue;

        auto thisClust = hh->cluster();
        float P_Charge = thisClust->charge();

        bool thisBad,thisEdge,thisBig;
        thisBig  = hh->spansTwoROCs();
        thisBad  = hh->hasBadPixels();
        thisEdge = hh->isOnEdge();

        if(p_charge[hitBin] !=dummy)
        {
          if(!isBad[hitBin] && thisBad) continue;
          if(!isBad[hitBin] && !thisBad && fabs(P_Charge)<fabs(p_charge[hitBin])) continue;
        }

        auto rangeIn = tpClust->equal_range(hit->firstClusterRef());

        //for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
        //kPdgs.push_back((*ip->second).pdgId());

        if(rangeIn.first!=rangeIn.second)
        {
          pdgId[i] = (double)((*rangeIn.first->second).pdgId());
          pdgIds[i] = (double)((*rangeIn.first->second).pdgId());
          // std::cout << pdgId[i] << std::endl;

          if((*rangeIn.first->second).genParticle_begin()!=(*rangeIn.first->second).genParticle_end())
            if((*(*rangeIn.first->second).genParticle_begin())->mother()!=nullptr)
              motherPdgId[i] = (double)((*(*rangeIn.first->second).genParticle_begin())->mother()->pdgId());

          if(pdgMomMap.find(motherPdgId[i]) != pdgMomMap.end())
            ++pdgMomMap[motherPdgId[i]];
          else
            pdgMomMap[motherPdgId[i]] = 1;

          if(pdgMap.find(pdgId[i]) != pdgMap.end())
            ++pdgMap[pdgId[i]];
          else
            pdgMap[pdgId[i]] = 1;

        }

        x[i] = (float)h->globalState().position.y();
        y[i] = (float)h->globalState().position.y();
        z[i] = hit_z;
        phi_hit[i] = (float)h->globalState().phi;
        r[i] = hit_r;
        n_seq[i] = (float)hitCounter;

        charge[i] = P_Charge;

        pdgId[i] = dummy;
        motherPdgId[i] = dummy;

        dZ[i] = (float)gDet->surface().bounds().thickness();
        ax1[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
        ax2[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
        ax3[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,1.,0.)).perp();
        ax4[i] = (float)gDet->surface().toGlobal(Local3DPoint(1.,0.,0.)).perp();
        rawId[i] = (float)gDet->geographicalId().rawId();

        p_size[i]  = (float)clust->size();
        p_sizex[i]  = (float)clust->sizeX();
        p_sizey[i]  = (float)clust->sizeY();
        p_x[i]  = (float)clust->x();
        p_y[i]  = (float)clust->y();
        p_ovx[i]  = (float)clust->sizeX() > 16.;
        p_ovy[i]  = (float)clust->sizeY() > 16.;
        p_skew[i]  = (float)clust->sizeY() / (float)clust->sizeX();
        p_big[i]  = thisBig;
        p_bad[i]  = thisBad;
        p_edge[i]  = thisEdge;
        p_charge[i] = P_Charge;

        TH2F hClust("hClust","hClust",
        padSize,
        clust->x()-padHalfSize,
        clust->x()+padHalfSize,
        padSize,
        clust->y()-padHalfSize,
        clust->y()+padHalfSize);

        //Initialization
        for (int nx = 0; nx < padSize; ++nx)
          for (int ny = 0; ny < padSize; ++ny)
            hClust.SetBinContent(nx,ny,0.0);

        for (int k = 0; k < clust->size(); ++k)
          hClust.SetBinContent(hClust.FindBin((float)clust->pixel(k).x, (float)clust->pixel(k).y),(float)clust->pixel(k).adc);

        int c = 0;
        for (int ny = padSize; ny>0; --ny)
        {
          for(int nx = 0; nx<padSize; nx++)
          {

            int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
            hitPixels[hitBin][c] = (float)hClust.GetBinContent(n);
            c++;
          }
        }

      }//if pix

      if(hitBin>9) //Strip Hit!
      {
        int stripBin = hitBin-10;
        const SiStripRecHit2D* siStripHit2D = dynamic_cast<SiStripRecHit2D const *>(hit);
        const SiStripRecHit1D* siStripHit1D = dynamic_cast<SiStripRecHit1D const *>(hit);

        if(!siStripHit2D && !siStripHit1D) continue;

        auto thisClust = h->firstClusterRef();

        float S_Charge = thisClust.charge();

        if(s_charge[stripBin] !=dummy && fabs(S_Charge)<fabs(s_charge[stripBin])) continue;

        auto rangeIn = tpClust->equal_range(h->firstClusterRef());

        //for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
        //kPdgs.push_back((*ip->second).pdgId());

        if(rangeIn.first!=rangeIn.second)
        {
          pdgId[i] = (double)((*rangeIn.first->second).pdgId());
          pdgIds[i] = (double)((*rangeIn.first->second).pdgId());
          // std::cout << pdgId[i] << std::endl;

          if((*rangeIn.first->second).genParticle_begin()!=(*rangeIn.first->second).genParticle_end())
            if((*(*rangeIn.first->second).genParticle_begin())->mother()!=nullptr)
              motherPdgId[i] = (double)((*(*rangeIn.first->second).genParticle_begin())->mother()->pdgId());

          if(pdgMomMap.find(motherPdgId[i]) != pdgMomMap.end())
            ++pdgMomMap[motherPdgId[i]];
          else
            pdgMomMap[motherPdgId[i]] = 1;

          if(pdgMap.find(pdgId[i]) != pdgMap.end())
            ++pdgMap[pdgId[i]];
          else
            pdgMap[pdgId[i]] = 1;

        }

        x[i] = (float)h->globalState().position.y();
        y[i] = (float)h->globalState().position.y();
        z[i] = hit_z;
        phi_hit[i] = (float)h->globalState().phi;
        r[i] = hit_r;
        n_seq[i] = (float)hitCounter;

        charge[i] = S_Charge;

        pdgId[i] = dummy;
        motherPdgId[i] = dummy;

        dZ[i] = (float)gDet->surface().bounds().thickness();
        ax1[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
        ax2[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
        ax3[i] = (float)gDet->surface().toGlobal(Local3DPoint(0.,1.,0.)).perp();
        ax4[i] = (float)gDet->surface().toGlobal(Local3DPoint(1.,0.,0.)).perp();
        rawId[i] = (float)gDet->geographicalId().rawId();

        if(siStripHit1D) s_dim[stripBin] = 1.0;
        if(siStripHit2D) s_dim[stripBin] = 2.0;


        s_center[stripBin] = (float)thisClust.barycenter();
        s_first[stripBin] = (float)thisClust.firstStrip();
        s_merged[stripBin] = (float)thisClust.isMerged();
        s_size[stripBin] = (float)thisClust.amplitudes().size();
        s_charge[stripBin] = (float)thisClust.charge();

        int minSize = -std::max(int(-thisClust->amplitudes().size()),-16);

        for(int j =0;j<16;j++)
          hitStrips[stripBin][j] = dummy;

        for(int j =0;j<minSize;j++)
          hitStrips[stripBin][j] = (float)thisClust->amplitudes()[j];

      }//if strip

    } //hit loop


    int allMatched = 0;
    trackPdg = 0.0;

    if(pdgMap.size()>0)
    {
      auto modePdg = std::max_element(pdgMap.begin(), pdgMap.end(),[](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {return p1.second < p2.second; });
      for (auto const& p : pdgIds)
          if(p.second==modePdg->first)
            ++allMatched;
      sharedFraction = (float)(float(allMatched)/float(nHits));
      // std::cout << tt << " - " << modePdg->first << " - " << sharedFraction << std::endl;
      trackPdg = modePdg->first;
    }
    else
    {
      trackPdg = 0.0;
      sharedFraction = 0.0;
      // std::cout << tt << " - UnMatched " << std::endl;
    }

    if(pdgMomMap.size()>0)
    {
      auto modePdg = std::max_element(pdgMomMap.begin(), pdgMomMap.end(),[](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {return p1.second < p2.second; });
      for (auto const& p : pdgIds)
          if(p.second==modePdg->first)
            ++allMatched;
      sharedMomFraction = (float)(float(allMatched)/float(nHits));
      // std::cout << tt << " - " << modePdg->first << " - " << sharedFraction << std::endl;
      trackMomPdg = modePdg->first;
    }
    else
    {
      trackMomPdg = 0.0;
      sharedMomFraction = 0.0;
      // std::cout << tt << " - UnMatched " << std::endl;
    }

    theData.push_back((double)trackPdg);
    theData.push_back((double)sharedFraction);
    theData.push_back((double)trackMomPdg);
    theData.push_back((double)sharedMomFraction);

    for(int i = 0; i<10;i++)
    {

      theData.push_back((double)x[i]);
      theData.push_back((double)y[i]);
      theData.push_back((double)z[i]);

      theData.push_back((double)phi_hit[i]);
      theData.push_back((double)r[i]);

      theData.push_back((double)c_x[i]);
      theData.push_back((double)c_y[i]);
      theData.push_back((double)size[i]);
      theData.push_back((double)sizex[i]);
      theData.push_back((double)sizey[i]);

      theData.push_back((double)charge[i]);

      theData.push_back((double)ovfx[i]);
      theData.push_back((double)ovfy[i]);

      theData.push_back((double)ratio[i]);

      theData.push_back((double)motherPdgId[i]);
      theData.push_back((double)pdgId[i]);

    }

    for(int i = 0; i<10;i++)
      for(int j =0;j<padSize*padSize;j++)
        theData.push_back((double)(hitPixels[i][j]));

    if(pdgMap.size()>0)
    {
      for (size_t i = 0; i < theData.size(); i++) {
        trackFile << theData[i] << "\t";
      }
      trackFile << 542.1369 << std::endl;

      //cnntree->Fill();
    }
  }

// std::cout << "Closing" << std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void
CNNTrackAnalyze::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
CNNTrackAnalyze::endJob()
{
  // std::cout << "Closing" << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNNTrackAnalyze::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNTrackAnalyze);
