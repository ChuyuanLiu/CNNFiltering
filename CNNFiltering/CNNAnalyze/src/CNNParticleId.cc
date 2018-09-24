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
#include "RecoTracker/TkHitPairs/interface/IntermediateHitfloatts.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitfloatts.h"

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

  int floattSize;
  std::string processName_;
  edm::EDGetTokenT<edm::View<reco::Track>> alltracks_;
  int minPix_;

  edm::EDGetTokenT<reco::BeamSpot>  bsSrc_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  // edm::GetterOfProducts<IntermediateHitfloatts> getterOfProducts_;

  float padHalfSize;
  int padSize, tParams;

  float pt, eta, phi, p, chi2n, d0, dx, dz, sharedFraction,sharedMomFraction;
  int nhit, nhpxf, nhtib, nhtob, nhtid, nhtec, nhpxb, nHits, trackPdg,trackMomPdg;
  int eveNumber, runNumber, lumNumber;

  std::vector<float>  x, y, z, phi_hit, r, c_x, c_y, charge, ovfx, ovfy;
  std::vector<float> ratio, pdgId, motherPdgId, size, sizex, sizey;
  //std::vector<TH2> hitClust;


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

  // std::cout<<"CNNfloatts Analyzer"<<std::endl;

  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality= reco::TrackBase::qualityByName(theTrackQuality);

  edm::Handle<View<reco::Track> >  trackCollection;
  iEvent.getByToken(alltracks_, trackCollection);

  eveNumber = iEvent.id().event();
  runNumber = iEvent.id().run();
  lumNumber = iEvent.id().luminosityBlock();

  std::string fileName = processName_ + ".txt";
  //std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
  //fileName += "_" + processName_ + "_dnn_floatts.txt";
  // std::ofstream outCNNFile(fileName, std::ofstream::app);

  std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31}; //seqNumbers of pixel detectors 0,1,2,3 barrel 14,15,16, fwd 29,30,31 bkw
  std::vector<int> partiList{11,13,15,22,111,211,311,321,2212,2112,3122,223};

  int numFeats = 188;
  int numTracks = trackCollection->size();

  tensorflow::Tensor inputFeat(tensorflow::DT_FLOAT, {numTracks,numFeats});
  float* vLab = inputFeat.flat<float>().data();
  std::vector<tensorflow::Tensor> outputs;

  for(edm::View<reco::Track>::size_type tt=0; tt<trackCollection->size(); ++tt)
  {
    int trackOffset = numFeats * tt;

    std::vector<float> theData;
    // std::cout << "Track ------------------- "<< std::endl;
    // std::cout << std::endl;
    std::map<int,const TrackerSingleRecHit*> theHits;
    std::map<int,bool> isBad,isEdge,isBig;
    std::map<int,float> hitSize,pdgIds,flagHit;
    std::map<float,int> pdgMap,pdgMomMap;

    auto track = trackCollection->refAt(tt);
    auto hitPattern = track->hitPattern();
    bool trkQual  = track->quality(trackQuality);

    float pId = hashId(0.1,0.2,0.3,0.4,0.5);

    // track->setPionId(0.2);
    // track->setMuonId(0.3);
    // track->setElecId(0.4);
    // track->setElseId(0.5);

    sharedFraction = 0.0;
    nHits = 0;


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

          track->setParticleId(0.0);std::cout << ".";

      // track->setMuonId(0.0);
      // track->setElecId(0.0);
      // track->setElseId(0.0);
      continue;
    }

    track->setParticleId(pId);std::cout << ".";

    vLab[iLab + trackOffset] =(float)track->pt();iLab++;
    vLab[iLab + trackOffset] =(float)track->eta();iLab++;
    vLab[iLab + trackOffset] =(float)track->phi();iLab++;
    vLab[iLab + trackOffset] =(float)track->p();iLab++;
    vLab[iLab + trackOffset] =(float)track->normalizedChi2();iLab++;
    vLab[iLab + trackOffset] =(float)track->numberOfValidHits();iLab++;
    vLab[iLab + trackOffset] =(float)track->d0();iLab++;
    vLab[iLab + trackOffset] =(float)track->dz();iLab++;

    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidPixelBarrelHits();iLab++;
    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidPixelEndcapHits();iLab++;
    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidStripTIBHits();iLab++;
    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidStripTOBHits();iLab++;
    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidStripTIDHits();iLab++;
    vLab[iLab + trackOffset] =(float)hitPattern.numberOfValidStripTECHits();iLab++;

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

        }else
          keepThis = true;

        if(keepThis)
          {
            theHits[hitLayer] = hit;
            hitSize[hitLayer] = (float)thisSize;
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

          x[i] = (float)h->globalState().position.y();
          y[i] = (float)h->globalState().position.y();
          z[i] = (float)h->globalState().position.z();
          phi_hit[i] = (float)h->globalState().phi;
          r[i] = (float)h->globalState().r;
          c_x[i] =(float)clust->x();
          c_y[i] =(float)clust->y();
          size[i] =(float)clust->size();
          sizex[i] =(float)clust->sizeX();
          sizey[i] =(float)clust->sizeY();
          charge[i] =(float)clust->charge();
          ovfx[i] =(float)clust->sizeX() > padSize;
          ovfy[i] =(float)clust->sizeY() > padSize;
          ratio[i] =(float)(clust->sizeY()) / (float)(clust->sizeX());


          auto rangeIn = tpClust->equal_range(h->firstClusterRef());

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

    vLab[iLab + trackOffset] =(float)trackPdg); iLab++;
    vLab[iLab + trackOffset] =(float)sharedFraction);iLab++;
    vLab[iLab + trackOffset] =(float)trackMomPdg);iLab++;
    vLab[iLab + trackOffset] =(float)sharedMomFraction);iLab++;

    for(int i = 0; i<10;i++)
    {

      vLab[iLab + trackOffset] =(float)x[i]; iLab++;
      vLab[iLab + trackOffset] =(float)y[i]; iLab++;
      vLab[iLab + trackOffset] =(float)z[i]; iLab++;

      vLab[iLab + trackOffset] =(float)phi_hit[i]; iLab++;
      vLab[iLab + trackOffset] =(float)r[i]; iLab++;

      vLab[iLab + trackOffset] =(float)c_x[i]; iLab++;
      vLab[iLab + trackOffset] =(float)c_y[i]; iLab++;
      vLab[iLab + trackOffset] =(float)size[i]; iLab++;
      vLab[iLab + trackOffset] =(float)sizex[i]; iLab++;
      vLab[iLab + trackOffset] =(float)sizey[i]; iLab++;

      vLab[iLab + trackOffset] =(float)charge[i]; iLab++;

      vLab[iLab + trackOffset] =(float)ovfx[i]; iLab++;
      vLab[iLab + trackOffset] =(float)ovfy[i]; iLab++;

      vLab[iLab + trackOffset] =(float)ratio[i]; iLab++;

      vLab[iLab + trackOffset] =(float)motherPdgId[i]; iLab++;
      vLab[iLab + trackOffset] =(float)pdgId[i]); iLab++;

    }

    std::cout << "iLab = "<< iLab << std::endl;


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
