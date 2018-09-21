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
genParticles_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
traParticles_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("traParticles"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap"))),
trMap_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getParameter<edm::InputTag>("trMap"))),
genMap_(consumes<reco::TrackToGenParticleAssociator>(iConfig.getParameter<edm::InputTag>("genMap")))
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

  // edm::Service<TFileService> fs;
  // cnntree = fs->make<TTree>("CNNTree","Doublets Tree");
  //
  // cnntree->Branch("eveNumber",      &eveNumber,          "eveNumber/I");
  // cnntree->Branch("runNumber",      &runNumber,          "runNumber/I");
  // cnntree->Branch("lumNumber",      &lumNumber,          "lumNumber/I");
  //
  // cnntree->Branch("pt",      &pt,          "pt/D");
  // cnntree->Branch("eta",     &eta,          "eta/D");
  // cnntree->Branch("phi",     &phi,          "phi/D");
  // cnntree->Branch("p",       &p,          "p/D");
  // cnntree->Branch("chi2n",   &chi2n,  "chi2n/D");
  // cnntree->Branch("d0",      &d0,          "d0/D");
  // cnntree->Branch("dx",      &dx,          "dx/D");
  // cnntree->Branch("dz",      &dz,          "dz/D");
  //
  // cnntree->Branch("sharedFraction",      &sharedFraction,          "sharedFraction/D");
  //
  // cnntree->Branch("nhit",      &nhit,            "nhit/I");
  // cnntree->Branch("nhpxf",      &nhpxf,          "nhpxf/I");
  // cnntree->Branch("nhtib",      &nhtib,          "nhtib/I");
  // cnntree->Branch("nhtob",      &nhtob,          "nhtob/I");
  // cnntree->Branch("nhtid",      &nhtid,          "nhtid/I");
  // cnntree->Branch("nhtec",      &nhtec,          "nhtec/I");
  // cnntree->Branch("nhpxb",      &nhpxb,          "nhpxb/I");
  // cnntree->Branch("nhpxb",      &nhpxb,          "nhpxb/I");
  // cnntree->Branch("nHits",      &nHits,          "nHits/I");
  //
  // cnntree->Branch("trackPdg",      &trackPdg,          "trackPdg/I");
  //
  //
  // for(int i = 0; i<10;i++)
  // {
  //   std::string name,tree;
  //
  //   for(int j = 0;j<padSize*padSize;j++)
  //   {
  //     name = "hit_" + std::to_string(i) + "_Pix_" + std::to_string(j);
  //     tree = name + "/D";
  //     cnntree->Branch(name.c_str(),      &hitPixels[i][j],          tree.c_str());
  //   }
  //
  //   name = "hit_" + std::to_string(i) + "_x"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &x[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_y"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &y[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_z"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &z[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_phi_hit"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &phi_hit[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_r"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &r[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_c_x"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &c_x[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_c_y"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &c_y[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_pdgId"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &pdgId[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_motherPdgId"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &motherPdgId[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_size"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &size[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_sizex"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &sizex[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_sizey"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &sizey[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_charge"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &charge[i],          tree.c_str());
  //
  //   name = "hit_" + std::to_string(i) + "_ovfx"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &ovfx[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_ovfy"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &ovfy[i],          tree.c_str());
  //   name = "hit_" + std::to_string(i) + "_ratio"; tree = name + "/D";
  //   cnntree->Branch(name.c_str(),      &ratio[i],          tree.c_str());
  //
  //
  // }


  edm::InputTag beamSpotTag = iConfig.getParameter<edm::InputTag>("beamSpot");
  bsSrc_ = consumes<reco::BeamSpot>(beamSpotTag);

  infoPileUp_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter< edm::InputTag >("infoPileUp"));



}


CNNParticleId::~CNNParticleId()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

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

  edm::Handle<ClusterTPAssociation> tpClust;
  iEvent.getByToken(tpMap_,tpClust);

  edm::Handle<reco::TrackToTrackingParticleAssociator> tpTracks;
  iEvent.getByToken(trMap_, tpTracks);

  // edm::Handle<reco::GenParticleCollection>  genParticles ;
  // iEvent.getByToken(genParticles_,genParticles);

  // const reco::TrackToGenParticleAssociator* genTracks =nullptr;
  // edm::Handle<reco::TrackToGenParticleAssociator> trackGenAssociatorH;
  // iEvent.getByToken(genMap_,trackGenAssociatorH);
  // genTracks = trackGenAssociatorH.product();

  //Reco To GEN association
  // reco::RecoToGenCollection recGenColl;
  // recGenColl=genTracks->associateRecoToGen(trackCollection,genParticles);

  //Reco To SIM association
  // edm::RefToBaseVector<reco::Track> trackRefs;
  // for(edm::View<reco::Track>::size_type i=0; i<trackCollection->size(); ++i) {
  //   trackRefs.push_back(trackCollection->refAt(i));
  // }
  //
  // TrackingParticleRefVector tparVec;
  // const TrackingParticleRefVector *tparPtr = nullptr;
  // edm::Handle<TrackingParticleCollection> tparCollection;
  // iEvent.getByToken(traParticles_,tparCollection);
  // for(size_t i=0, size=tparCollection->size(); i<size; ++i) {
  //   tparVec.push_back(TrackingParticleRef(tparCollection, i));
  // }
  // tparPtr = &tparVec;
  // TrackingParticleRefVector const & tPartVector = *tparPtr;
  //
  // reco::RecoToSimCollection recSimCollL = std::move(tpTracks->associateRecoToSim(trackRefs, tPartVector));
  // reco::RecoToSimCollection const * recSimCollP = &recSimCollL;
  // reco::RecoToSimCollection const & recSimColl  = *recSimCollP;

  eveNumber = iEvent.id().event();
  runNumber = iEvent.id().run();
  lumNumber = iEvent.id().luminosityBlock();

  std::string fileName = processName_ + ".txt";
  //std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
  //fileName += "_" + processName_ + "_dnn_doublets.txt";
  std::ofstream outCNNFile(fileName, std::ofstream::app);

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

    track->setKaonId(0.1);
    track->setPionId(0.2);
    track->setMuonId(0.3);
    track->setElecId(0.4);
    track->setElseId(0.5);

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
    if(pixHits < 3)
      continue;

    // pt    = (double)track->pt();
    // eta   = (double)track->eta();
    // phi   = (double)track->phi();
    // p     = (double)track->p();
    // chi2n = (double)track->normalizedChi2();
    // nhit  = (double)track->numberOfValidHits();
    // d0    = (double)track->d0();
    // dz    = (double)track->dz();

    // nhpxb   = hitPattern.numberOfValidPixelBarrelHits();
    // nhpxf   = hitPattern.numberOfValidPixelEndcapHits();
    // nhtib   = hitPattern.numberOfValidStripTIBHits();
    // nhtob   = hitPattern.numberOfValidStripTOBHits();
    // nhtid   = hitPattern.numberOfValidStripTIDHits();
    // nhtec   = hitPattern.numberOfValidStripTECHits();

    // std::cout<<nhit<<std::endl;
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

    for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
    {
      TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(*recHit);

      if(!hit)
        continue;

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
          {
              if(!(thisBig || thisBad || thisEdge))
                keepThis = true;
              else
              {
                if(thisSize > hitSize[hitLayer])
                keepThis = true;
              }
          }
          else
          {
            if(!(thisBig || thisBad || thisEdge))
            {
              if(thisSize > hitSize[hitLayer])
              keepThis = true;
            }
          }
          //   if(!(thisBad || thisEdge || thisBig))
          //     keepThis = true;
          // if(isBad[hitLayer] && !thisBad)
          //     keepThis = true;
          // if((isBad[hitLayer] && thisBad)||(!(isBad[hitLayer] || thisBad)))
          //   if(thisSize > hitSize[hitLayer])
          //     keepThis = true;
          // if(thisBad && !isBad[hitLayer])
          //     keepThis = false;
        }else
          keepThis = true;

        if(keepThis)
          {
            theHits[hitLayer] = hit;
            hitSize[hitLayer] = (double)thisSize;
            isBad[hitLayer] = thisBad;
            isEdge[hitLayer] = thisEdge;
            isBig[hitLayer] = thisBad;
          }
        flagHit[hitLayer] = 1.0;

      }


    }

    for(int i = 0; i<10;i++)
    {
        if(theHits.find(i) != theHits.end())
        {
          ++nHits;
          auto h = theHits[i];

          // std::cout << h->geographicalId().subdetId() << '\n';

          const SiPixelRecHit* pixHit = dynamic_cast<SiPixelRecHit const *>(h);

          //auto rangeIn = tpClust->equal_range(bhit->firstClusterRef());
          auto clust = pixHit->cluster();

          x[i] = (double)h->globalState().position.y();
          y[i] = (double)h->globalState().position.y();
          z[i] = (double)h->globalState().position.z();
          phi_hit[i] = (double)h->globalState().phi;
          r[i] = (double)h->globalState().r;
          c_x[i] =(double)clust->x();
          c_y[i] =(double)clust->y();
          size[i] =(double)clust->size();
          sizex[i] =(double)clust->sizeX();
          sizey[i] =(double)clust->sizeY();
          charge[i] =(double)clust->charge();
          ovfx[i] =(double)clust->sizeX() > padSize;
          ovfy[i] =(double)clust->sizeY() > padSize;
          ratio[i] =(double)(clust->sizeY()) / (double)(clust->sizeX());

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


          int c = 0;
          for (int ny = padSize; ny>0; --ny)
          {
            for(int nx = 0; nx<padSize; nx++)
            {

              int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
              hitPixels[i][c] = (double)hClust.GetBinContent(n);
              c++;
            }
          }

        }
    }

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
        outCNNFile << theData[i] << "\t";
      }
      outCNNFile << 542.1369 << std::endl;

      //cnntree->Fill();
    }
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
