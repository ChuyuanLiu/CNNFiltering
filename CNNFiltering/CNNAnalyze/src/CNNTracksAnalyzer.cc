// -*- C++ -*-
//
// Package:    CNNFiltering/CNNTracksAnalyze
// Class:      CNNTracksAnalyze
//
/**\class CNNTracksAnalyze CNNTracksAnalyze.cc CNNFiltering/CNNTracksAnalyze/plugins/CNNTracksAnalyze.cc

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

class CNNTracksAnalyze : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit CNNTracksAnalyze(const edm::ParameterSet&);
  ~CNNTracksAnalyze();

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
  std::vector<std::string>  HLTs_;
  // edm::GetterOfProducts<IntermediateHitDoublets> getterOfProducts_;

  int nevents, ntracks;

  UInt_t    run;
  ULong64_t event;
  UInt_t    lumiblock;
  UInt_t    pu;

  Double_t pt, eta, phi, pdg, mompdg;
  Double_t charge, dxy, dz, nHits, NPixelHits, NStripHits, NTrackhits;
  Double_t NBPixHits, NPixLayers, NTraLayers, NStrLayers, NBPixLayers;

  std::array<double,20> hltword;
  std::array< std::array <double,8>, 25 > hitCoords;
  std::array< std::array <double,13>, 25 > pixelInfos;
  std::array< std::array <double,256>, 25 > pixelADC;
  // std::array< std::array <double,20>, 25 > pixelADC;
  // std::array< std::array <double,20>, 25 > pixelADCx;
  // std::array< std::array <double,20>, 25 > pixelADCy;
  std::array< std::array <double,7>, 25 > stripInfos;
  std::array< std::array <double,20>, 25 > stripADC;


  TTree *track_tree;


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
CNNTracksAnalyze::CNNTracksAnalyze(const edm::ParameterSet& iConfig):
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap"))),
// TriggerResults_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResults"))),
// HLTs_(iConfig.getParameter<std::vector<std::string>>("HLTs"))
{



  edm::InputTag beamSpotTag = iConfig.getParameter<edm::InputTag>("beamSpot");
  bsSrc_ = consumes<reco::BeamSpot>(beamSpotTag);

  infoPileUp_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter< edm::InputTag >("infoPileUp"));



}


CNNTracksAnalyze::~CNNTracksAnalyze()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
CNNTracksAnalyze::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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


  float padHalfSize = 8.0;
  int padSize = 16;
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

  std::string fileName = std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber) + "_cnn_tracks.txt";
  ;
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
    std::vector<float> theData;
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
      rawId[i] = 0.0;

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

    // pt    = (float)track->pt();
    // eta   = (float)track->eta();
    // phi   = (float)track->phi();
    // p     = (float)track->p();
    // chi2n = (float)track->normalizedChi2();
    // nhit  = (float)track->numberOfValidHits();
    // d0    = (float)track->d0();
    // dz    = (float)track->dz();

    // nhpxb   = hitPattern.numberOfValidPixelBarrelHits();
    // nhpxf   = hitPattern.numberOfValidPixelEndcapHits();
    // nhtib   = hitPattern.numberOfValidStripTIBHits();
    // nhtob   = hitPattern.numberOfValidStripTOBHits();
    // nhtid   = hitPattern.numberOfValidStripTIDHits();
    // nhtec   = hitPattern.numberOfValidStripTECHits();

    // std::cout<<nhit<<std::endl;
    theData.push_back((float)eveNumber);
    theData.push_back((float)runNumber);
    theData.push_back((float)lumNumber);

    theData.push_back((float)track->pt());
    theData.push_back((float)track->eta());
    theData.push_back((float)track->phi());
    theData.push_back((float)track->p());
    theData.push_back((float)track->normalizedChi2());
    theData.push_back((float)track->numberOfValidHits());
    theData.push_back((float)track->d0());
    theData.push_back((float)track->dz());

    theData.push_back((float)hitPattern.numberOfValidPixelHits());
    theData.push_back((float)hitPattern.numberOfValidStripHits());
    theData.push_back((float)hitPattern.numberOfValidTrackerHits());
    theData.push_back((float)hitPattern.numberOfValidStripHits());

    theData.push_back((float)hitPattern.pixelLayersWithMeasurement());
    theData.push_back((float)hitPattern.trackerLayersWithMeasurement());
    theData.push_back((float)hitPattern.stripLayersWithMeasurement());
    theData.push_back((float)hitPattern.pixelBarrelLayersWithMeasurement());

    theData.push_back((float)hitPattern.numberOfValidPixelBarrelHits());
    theData.push_back((float)hitPattern.numberOfValidPixelEndcapHits());
    theData.push_back((float)hitPattern.numberOfValidStripTIBHits());
    theData.push_back((float)hitPattern.numberOfValidStripTOBHits());
    theData.push_back((float)hitPattern.numberOfValidStripTIDHits());
    theData.push_back((float)hitPattern.numberOfValidStripTECHits());

    std::vector < std::array <float,10> >  hitCoords_;
    //n x y z r phi ax1 ax2
    std::vector < std::array <float,13> > pixelInfos_;
    //c_x c_y size size_x size_y charge ovfx ovfy ratio isBig isBad isOnEdge id
    std::vector < std::array <float,20> > pixelADC_, pixelADCx_, pixelADCy_;

    std::vector < std::array <float,7> > stripInfos_;
    //dimension first ampsize charge merged splitClusterError id
    std::vector < std::array <float,20> > stripADC_;

    for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
    {
      TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(*recHit);

      if(!hit)
        continue;

      DetId detId = (*recHit)->geographicalId();
      unsigned int subdetid = detId.subdetId();
      if(detId.det() != DetId::Tracker) continue;

      const GeomDet* gDet = (hit)->det();
      float dZ = gDet->surface().bounds().thickness();
      float x = (hit)->globalState().position.x();
      float y = (hit)->globalState().position.y();
      float z = (hit)->globalState().position.z();
      float phi = (hit)->globalState().phi;
      float r = (hit)->globalState().r;

      float ax1 = gDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
      float ax2 = gDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
      float ax3 = gDet->surface().toGlobal(Local3DPoint(0.,1.,0.)).perp();
      float ax4 = gDet->surface().toGlobal(Local3DPoint(1.,0.,0.)).perp();

      float rawId = gDet->geographicalId().rawId();

      hitCoords_.push_back(std::array<float,10> {{float(n),x,y,z,phi,r,ax1,ax2,dZ,rawId}});


      SiPixelRecHit *pixHit = dynamic_cast<SiPixelRecHit*>(hit);
      if(pixHit)
      {

        std::array<float,13> pixInf;
        auto clust = pixHit->cluster();


        auto rangeIn = tpClust->equal_range(pixHit->firstClusterRef());

        //for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
        //kPdgs.push_back((*ip->second).pdgId());

        if(rangeIn.first!=rangeIn.second)
          {
            pdgId[i] = (float)((*rangeIn.first->second).pdgId());
            pdgIds[i] = (float)((*rangeIn.first->second).pdgId());
            // std::cout << pdgId[i] << std::endl;

            if((*rangeIn.first->second).genParticle_begin()!=(*rangeIn.first->second).genParticle_end())
              if((*(*rangeIn.first->second).genParticle_begin())->mother()!=nullptr)
                motherPdgId[i] = (float)((*(*rangeIn.first->second).genParticle_begin())->mother()->pdgId());

            if(pdgMomMap.find(motherPdgId[i]) != pdgMomMap.end())
              ++pdgMomMap[motherPdgId[i]];
            else
              pdgMomMap[motherPdgId[i]] = 1;

            if(pdgMap.find(pdgId[i]) != pdgMap.end())
              ++pdgMap[pdgId[i]];
            else
              pdgMap[pdgId[i]] = 1;

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

        pixelInfos_.push_back(pixInf);

        std::array <float,20> pixADC, pixADCx, pixADCy;

        int minSize = -std::max(int(-clust->size()),-20);
        for (int k = 0; k<20; ++k)
        {
          pixADC[k]  = -1.0;
          pixADCx[k] = -1.0;
          pixADCy[k] = -1.0;
        }
        for (int k = 0; k < minSize; ++k)
        {
          pixADC[k]  = (float)clust->pixel(k).adc;
          pixADCx[k] = (float)clust->pixel(k).x;
          pixADCy[k] = (float)clust->pixel(k).y;

        }

  pixelADC_.push_back(pixADC);
  pixelADCx_.push_back(pixADCx);
  pixelADCy_.push_back(pixADCy);

        std::cout << " Si Pixel Rec Hit" << std::endl;

        n++;
        continue;
      }

      const SiStripRecHit2D* siStripHit2D = dynamic_cast<SiStripRecHit2D const *>(hit);
      if(siStripHit2D)
      {
        //dimension first ampsize charge merged splitClusterError id
        std::array<float,7> stripInf;

        auto clust = siStripHit2D->cluster();


        auto rangeIn = tpClust->equal_range(siStripHit2D->firstClusterRef());

        //for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
        //kPdgs.push_back((*ip->second).pdgId());

        if(rangeIn.first!=rangeIn.second)
          {
            pdgId[i] = (float)((*rangeIn.first->second).pdgId());
            pdgIds[i] = (float)((*rangeIn.first->second).pdgId());
            // std::cout << pdgId[i] << std::endl;

            if((*rangeIn.first->second).genParticle_begin()!=(*rangeIn.first->second).genParticle_end())
              if((*(*rangeIn.first->second).genParticle_begin())->mother()!=nullptr)
                motherPdgId[i] = (float)((*(*rangeIn.first->second).genParticle_begin())->mother()->pdgId());

            if(pdgMomMap.find(motherPdgId[i]) != pdgMomMap.end())
              ++pdgMomMap[motherPdgId[i]];
            else
              pdgMomMap[motherPdgId[i]] = 1;

            if(pdgMap.find(pdgId[i]) != pdgMap.end())
              ++pdgMap[pdgId[i]];
            else
              pdgMap[pdgId[i]] = 1;

          }


        stripInf[0] = float(n);
        stripInf[1] = 2.0;

        stripInf[2] = clust->charge();
        stripInf[3] = clust->barycenter();
        stripInf[4] = clust->firstStrip();
        stripInf[5] = clust->isMerged();
        stripInf[6] = clust->amplitudes().size();
        stripInfos_.push_back(stripInf);

        int minSize = -std::max(-int(clust->amplitudes().size()),-20);

        std::array <float,20> stripADC;

        for (int k = 0; k<20; ++k)
          stripADC[k]  = -1.0;
        for (int k = 0; k < minSize; ++k)
          stripADC[k]  = (float)clust->amplitudes()[k];

  stripADC_.push_back(stripADC);
        continue;
      }
      const SiStripRecHit1D* siStripHit1D = dynamic_cast<SiStripRecHit1D const *>(hit);
      if(siStripHit1D)
      {
        //dimension first ampsize charge merged splitClusterError id
        std::array<float,7> stripInf;

        auto clust = siStripHit1D->cluster();

        auto rangeIn = tpClust->equal_range(siStripHit1D->firstClusterRef());

        //for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
        //kPdgs.push_back((*ip->second).pdgId());

        if(rangeIn.first!=rangeIn.second)
          {
            pdgId[i] = (float)((*rangeIn.first->second).pdgId());
            pdgIds[i] = (float)((*rangeIn.first->second).pdgId());
            // std::cout << pdgId[i] << std::endl;

            if((*rangeIn.first->second).genParticle_begin()!=(*rangeIn.first->second).genParticle_end())
              if((*(*rangeIn.first->second).genParticle_begin())->mother()!=nullptr)
                motherPdgId[i] = (float)((*(*rangeIn.first->second).genParticle_begin())->mother()->pdgId());

            if(pdgMomMap.find(motherPdgId[i]) != pdgMomMap.end())
              ++pdgMomMap[motherPdgId[i]];
            else
              pdgMomMap[motherPdgId[i]] = 1;

            if(pdgMap.find(pdgId[i]) != pdgMap.end())
              ++pdgMap[pdgId[i]];
            else
              pdgMap[pdgId[i]] = 1;

          }

        stripInf[0] = float(n);
        stripInf[1] = 1.0;

        stripInf[2] = clust->charge();
        stripInf[3] = clust->barycenter();
        stripInf[4] = clust->firstStrip();
        stripInf[5] = clust->isMerged();
        stripInf[6] = clust->amplitudes().size();
        stripInfos_.push_back(stripInf);

        int minSize = -std::max(int(-clust->amplitudes().size()),-20);

  std::array <float,20> stripADC;

        for (int k = 0; k<20; ++k)
          stripADC[k]  = -1.0;
        for (int k = 0; k < minSize; ++k)
          stripADC[k]  = (float)clust->amplitudes()[k];

    stripADC_.push_back(stripADC);
  continue;
      }
    }

    std::array< std::array <double,8>, 25 > hitCoords;
    std::array< std::array <double,13>, 25 > pixelInfos;
    std::array< std::array <double,256>, 25 > pixelADC;
    // std::array< std::array <double,20>, 25 > pixelADC;
    // std::array< std::array <double,20>, 25 > pixelADCx;
    // std::array< std::array <double,20>, 25 > pixelADCy;
    std::array< std::array <double,7>, 25 > stripInfos;
    std::array< std::array <double,20>, 25 > stripADC;

    for (size_t j = 0; j < 25; j++) {
      for (size_t i = 0; i < 8; i++)
      {
        hitCoords[j][i] = -0.000012345678;
      }
      for (size_t i = 0; i < 13; i++)
      {
        pixelInfos[j][i] = -1.2345678;
      }

      for (size_t i = 0; i < 20; i++)
      {
        // pixelADCx[j][i] = -1.2345678;
        // pixelADCy[j][i] = -1.2345678;
        stripADC[j][i] = -1.2345678;
      }
      for (size_t i = 0; i < 256; i++)
      {
        pixelADC[j][i] = -1.2345678;
      }
      for (size_t i = 0; i < 7; i++)
      {
        stripInfos[j][i] = -1.2345678;
      }
    }

    int maxHits = 25;

    int noHits = hitCoords_.size();
    int minHits = -std::max(-maxHits,-noHits);

    for(int j = 0; j<minHits;j++)
    {
      auto coords  = hitCoords_[j];
      for (size_t i = 0; i < 8; i++)
      {
        hitCoords[j][i] = (Double_t) coords[i];
      }
    }

    for(int j = 0; j<25;j++)
    {
      for (size_t i = 0; i < 8; i++)
      {
        theData.push_back((float)hitCoords[j][i]);
      }
    }

    noHits = pixelInfos_.size();
    minHits = -std::max(-10,-noHits);
    ntracks++;

    for(int j = 0; j<minHits;j++)
    {
      auto pixinf  = pixelInfos_[j];
      auto pixadc  = pixelADC_[j];
      auto pixadx  = pixelADCx_[j];
      auto pixady  = pixelADCy_[j];

      TH2F hPixel("hPixel","hPixel",
      padSize,
      pixinf[1]-padHalfSize,
      pixinf[1]+padHalfSize,
      padSize,
      pixinf[2]-padHalfSize,
      pixinf[2]+padHalfSize);

      for (int nx = 0; nx < padSize; ++nx)
      {
        for (int ny = 0; ny < padSize; ++ny)
        {
          hPixel.SetBinContent(nx,ny,0.0);
        }
      }

      // std::cout << "Hist limits: " <<  pixinf[1]-padHalfSize << " - " <<  pixinf[1]+padHalfSize;
      // std::cout << " - " <<  pixinf[2]-padHalfSize << " - " <<  pixinf[2]+padHalfSize << std::endl;
      for (int k = 0; k < 20; ++k)
      {
        hPixel.SetBinContent(hPixel.FindBin(pixadx[k], pixady[k]),pixadc[k]);
        // std::cout << "Pixel "<< j << " " << k << " : " <<pixadx[k] << " - " << pixady[k] << " - " << pixadc[k] << std::endl;
      }

      int c = 0;
      for (int ny = padSize; ny>0; --ny)
      {
        for(int nx = 0; nx<padSize; nx++)
        {

          int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
          pixelADC[j][c] = (float)hPixel.GetBinContent(n);
          // std::cout << c << " " << hPixel.GetBinContent(n) << " " << std::endl;
          c++;
        }
      }


      for (size_t i = 0; i < 13; i++)
      {
        pixelInfos[j][i] = (Double_t) pixinf[i];
      }


    }

    for(int j = 0; j<10;j++)
    {
      for (size_t i = 0; i < 13; i++)
      {
        theData.push_back((float)pixelInfos[j][i]);
      }

      for (size_t i = 0; i < 256; i++)
      {
        theData.push_back((float)pixelADC[j][i]);
      }
    }


    noHits = stripInfos_.size();
    minHits = -std::max(-15,-noHits);
    ntracks++;

    for(int j = 0; j<minHits;j++)
    {
      auto strinf  = stripInfos_[j];
      auto stradc  = stripADC_[j];

      for (size_t i = 0; i < 20; i++)
      {

        stripADC[j][i] = (Double_t) stradc[i];
      }

      for (size_t i = 0; i < 7; i++)
      {
        stripInfos[j][i] = (Double_t) strinf[i];
      }
    }

    for(int j = 0; j<15;j++)
    {
      for (size_t i = 0; i < 13; i++)
      {
        theData.push_back((float)stripInfos[j][i]);
      }

      for (size_t i = 0; i < 256; i++)
      {
        theData.push_back((float)stripADC[j][i]);
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

    theData.push_back((float)trackPdg);
    theData.push_back((float)sharedFraction);
    theData.push_back((float)trackMomPdg);
    theData.push_back((float)sharedMomFraction);

      for (size_t i = 0; i < theData.size(); i++) {
        outCNNFile << theData[i] << "\t";
      }
      outCNNFile << 542.1369 << std::endl;

  }

// std::cout << "Closing" << std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void
CNNTracksAnalyze::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
CNNTracksAnalyze::endJob()
{
  // std::cout << "Closing" << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNNTracksAnalyze::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNTracksAnalyze);
