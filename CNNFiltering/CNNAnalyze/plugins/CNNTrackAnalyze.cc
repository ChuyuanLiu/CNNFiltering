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
  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
  edm::EDGetTokenT<reco::BeamSpot>  bsSrc_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  // edm::GetterOfProducts<IntermediateHitDoublets> getterOfProducts_;

  float padHalfSize;
  int padSize, tParams;

  float pt, eta, phi, p, chi2n, d0, dx, dz;
  int nhit, nhpxf, nhtib, nhtob, nhtid, nhtec, nhpxb;
  int eveNumber, runNumber, lumNumber;

  std::vector<float>  x, y, z, phi_hit, r, c_x, c_y, size, sizex, sizey, charge, ovfx, ovfy, ratio;
  //std::vector<TH2> hitClust;

  std::vector<float> hitPixel0, hitPixel1, hitPixel2, hitPixel3, hitPixel4;
  std::vector<float> hitPixel5, hitPixel6, hitPixel7, hitPixel8, hitPixel9;

  std::vector< std::vector<float> > hitPixels;

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
CNNTrackAnalyze::CNNTrackAnalyze(const edm::ParameterSet& iConfig):
processName_(iConfig.getParameter<std::string>("processName")),
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap")))
{

  padHalfSize = 7.5;
  padSize = (int)(padHalfSize*2);
  tParams = 26;

  hitPixels.reserve(10);

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

  x.reserve(10);
  edm::Service<TFileService> fs;
  cnntree = fs->make<TTree>("CNNTree","Doublets Tree");

  cnntree->Branch("eveNumber",      &eveNumber,          "eveNumber/I");
  cnntree->Branch("runNumber",      &runNumber,          "runNumber/I");
  cnntree->Branch("lumNumber",      &lumNumber,          "lumNumber/I");

  cnntree->Branch("pt",      &pt,          "pt/D");
  cnntree->Branch("eta",      &eta,          "eta/D");
  cnntree->Branch("phi",      &phi,          "phi/D");
  cnntree->Branch("p",      &p,          "p/D");
  cnntree->Branch("chi2n",      &chi2n,  "chi2n/D");
  cnntree->Branch("d0",      &d0,          "d0/D");
  cnntree->Branch("dx",      &dx,          "dx/D");
  cnntree->Branch("dz",      &dz,          "dz/D");

  cnntree->Branch("nhit",      &nhit,            "nhit/I");
  cnntree->Branch("nhpxf",      &nhpxf,          "nhpxf/I");
  cnntree->Branch("nhtib",      &nhtib,          "nhtib/I");
  cnntree->Branch("nhtob",      &nhtob,          "nhtob/I");
  cnntree->Branch("nhtid",      &nhtid,          "nhtid/I");
  cnntree->Branch("nhtec",      &nhtec,          "nhtec/I");
  cnntree->Branch("nhpxb",      &nhpxb,          "nhpxb/I");

  cnntree->Branch("nhpxb",      &nhpxb,          "nhpxb/I");

  for(int i = 0; i<10;i++)
  {
    std::string name = "hitPix_" + std::to_string(i);
    std::string tree = name + "/D";
    cnntree->Branch(name.c_str(),      &hitPixels[i],          tree.c_str());
  }

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

  // test = iEvent.id().event();
  //
  // cnntree->Fill();

  eveNumber = iEvent.id().event();
  runNumber = iEvent.id().run();
  lumNumber = iEvent.id().luminosityBlock();

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
    bool trkQual  = track->quality(trackQuality);

    if(!trkQual)
      continue;

    int pixHits = hitPattern.numberOfValidPixelHits();



    if(pixHits < 4)
      continue;

    pt    = track->pt();
    eta   = track->eta();
    phi   = track->phi();
    p     = track->p();
    chi2n = track->normalizedChi2();
    nhit  = track->numberOfValidHits();
    d0    = track->d0();
    dz    = track->dz();

    nhpxb   = hitPattern.numberOfValidPixelBarrelHits();
    nhpxf   = hitPattern.numberOfValidPixelEndcapHits();
    nhtib   = hitPattern.numberOfValidStripTIBHits();
    nhtob   = hitPattern.numberOfValidStripTOBHits();
    nhtid   = hitPattern.numberOfValidStripTIDHits();
    nhtec   = hitPattern.numberOfValidStripTECHits();

    std::cout<<nhit<<std::endl;

    for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
    {
      std::cout <<"Hit"<<std::endl;
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


    for(int i = 0; i<10;i++)
    {
        if(theHits.find(i) != theHits.end())
        {
          auto h = theHits[i];

          std::cout << h->geographicalId().subdetId() << '\n';

          const SiPixelRecHit* pixHit = dynamic_cast<SiPixelRecHit const *>(h);
          auto clust = pixHit->cluster();

          x.push_back((h->globalState()).position.y()); //1
          y.push_back((h->globalState()).position.y()); //1
          z.push_back((h->globalState()).position.z()); //1

        //
        //   (hit->globalState()).phi; //Phi //FIXME
        //   (hit->globalState()).r;
        // //ClusterInformations
        //   (float)clust->x(); //20
        //   (float)clust->y();
        //   (float)clust->size();
        //   (float)clust->sizeX();
        //   (float)clust->sizeY();
        //   (float)clust->pixel(0).adc; //25
        //   float(clust->charge())/float(clust->size()); //avg pixel charge
        //
        //
        //   (float)(clust->sizeX() > padSize);//27
        //   (float)(clust->sizeY() > padSize);
        //   (float)(clust->sizeY()) / (float)(clust->sizeX());
        //
        //
        //   (float)pixHit->spansTwoROCs();
        //   (float)pixHit->hasBadPixels();
        //   (float)pixHit->isOnEdge(); //31

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



          // //Linearizing the cluster
          //
          for (int ny = padSize; ny>0; --ny)
          {
            for(int nx = 0; nx<padSize; nx++)
            {
              int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
              hitPixels[i].push_back(hClust.GetBinContent(n));
            }
          }
          //
          // //ADC sum
          // thisHitPars.push_back(float(clust->charge()));

        }

    }

    std::cout<< "Filling" <<std::endl;

    cnntree->Fill();

  }

std::cout << "Closing" << std::endl;

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
  std::cout << "Closing" << std::endl;
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
