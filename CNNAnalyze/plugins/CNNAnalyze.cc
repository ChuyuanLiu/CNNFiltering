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

// user include files
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

#include <iostream>
#include <string>
#include <fstream>
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

        // ----------member data ---------------------------

      int doubletSize;
      edm::EDGetTokenT<IntermediateHitDoublets> intHitDoublets_;
      edm::EDGetTokenT<ClusterTPAssociation> tpMap_;


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
intHitDoublets_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap")))
{
   //now do what ever initialization is needed
   consumesMany<IntermediateHitDoublets>();
   usesResource("TFileService");

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

   int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
   std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

   std::cout<<"CNNDoublets Analyzer"<<std::endl;
   edm::Handle<IntermediateHitDoublets> iHd;
   iEvent.getByToken(intHitDoublets_,iHd);

   edm::Handle<ClusterTPAssociation> tpClust;
   iEvent.getByToken(tpMap_,tpClust);

   std::vector<edm::Handle<IntermediateHitDoublets> > intDoublets;
   iEvent.getManyByType(intDoublets);

   std::cout <<"No. of intdoubltes collections: " << intDoublets.size()<< std::endl;

   std::string fileName = "test.txt";
   std::ofstream test(fileName, std::ofstream::app);

   test << tpClust->size()  << std::endl;
   test << iHd->regionSize()  << std::endl;

   float padHalfSize = 7.5;
   int padSize = (int)(padHalfSize*2);

   for (std::vector<IntermediateHitDoublets::LayerPairHitDoublets>::const_iterator lIt= iHd->layerSetsBegin(); lIt != iHd->layerSetsEnd(); ++lIt)
   {
//     HitDoublets lDoublets = std::move(lIt->doublets());
     std::cout << "Size: " << lIt->doublets().size() << std::endl;
     for (size_t i = 0; i < lIt->doublets().size(); i++)
     {
              int inId = lIt->doublets().innerHitId(i);
              int outId = lIt->doublets().outerHitId(i);
              DetLayer const * innerLayer = detLayer(HitDoublets::inner);
              DetLayer const * outerLayer = detLayer(HitDoublets::outer);

              RecHitsSortedInPhi::Hit innerHit = lIt->doublets().hit(i, HitDoublets::inner);
              RecHitsSortedInPhi::Hit outerHit = lIt->doublets().hit(i, HitDoublets::outer);

              int detSeqIn = innerLayer->seqNum();
              int detSeqOut = outerLayer->seqNum();

              std::vector<int>::iterator detOnItOne = find(detOn.begin(),detOn.end(),innerLayer.detLayer()->seqNum());
              std::vector<int>::iterator detOnItTwo = find(detOn.begin(),detOn.end(),outerLayer.detLayer()->seqNum());

              auto rangeIn = tpClust->equal_range(innerHit->firstClusterRef());
              auto rangeOut = tpClust->equal_range(outerHit->firstClusterRef());
              std::cout << "Doublet no. "  << i << " hit no. " << inId << std::endl;
              // if(range.first == tpClust->end())
              //   std::cout << "No TP Matched "<<std::endl;
              for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
              {
		              // const auto tpKey = ip->second.key();
                  const auto tpPdgId = (*ip->second).pdgId();
                  std::cout << "Inner " << ip->second.key() << " - "<< tpPdgId  << std::endl;
              }

              for(auto ip=rangeOut.first; ip != rangeOut.second; ++ip)
              {
		              // const auto tpKey = ip->second.key();
                  const auto tpPdgId = (*ip->second).pdgId();
                  std::cout << "Outer " << ip->second.key() << " - "<< tpPdgId  << std::endl;
              }
     }
   }
   // auto range = clusterToTPMap.equal_range(dynamic_cast<const BaseTrackerRecHit&>(hit).firstClusterRef());
   //      for(auto ip=range.first; ip != range.second; ++ip) {
   //        const auto tpKey = ip->second.key();
   //        if(tpKeyToIndex.find(tpKey) == tpKeyToIndex.end()) // filter out TPs not given as an input
   //          continue;
   //        func(tpKey);
   //      }


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
