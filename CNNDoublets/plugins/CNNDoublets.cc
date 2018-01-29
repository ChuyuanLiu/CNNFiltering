// -*- C++ -*-
//
// Package:    CNNFiltering/CNNDoublets
// Class:      CNNDoublets
//
/**\class CNNDoublets CNNDoublets.cc CNNFiltering/CNNDoublets/plugins/CNNDoublets.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Adriano Di Florio
//         Created:  Mon, 29 Jan 2018 15:29:50 GMT
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

//
// class declaration
//

class CNNDoublets : public edm::stream::EDProducer<> {
   public:
      explicit CNNDoublets(const edm::ParameterSet&);
      ~CNNDoublets();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      int doubletSize;
      edm::EDGetTokenT<IntermediateHitDoublets> intHitDoublets_;
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
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
CNNDoublets::CNNDoublets(const edm::ParameterSet& iConfig):
intHitDoublets_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap")))
{

  produces<IntermediateHitDoublets>("newProdDoublets");
  doubletSize = 0;

}


CNNDoublets::~CNNDoublets()
{

   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CNNDoublets::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   std::cout<<"CNNDoublets Producer"<<std::endl;
   edm::Handle<IntermediateHitDoublets> iHd;
   iEvent.getByToken(intHitDoublets_,iHd);

   edm::Handle<ClusterTPAssociation> tpClust;
   iEvent.getByToken(tpMap_,tpClust);

   std::unique_ptr<IntermediateHitDoublets> oiHd(new IntermediateHitDoublets);
   std::cout<<*iHd->regionSize()<<std::endl;
   iEvent.put(std::move(oiHd),"newProdDoublets");

/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which
   // is put into the Event
   iEvent.put(std::make_unique<ExampleData2>(*pIn));
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/

}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CNNDoublets::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CNNDoublets::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
CNNDoublets::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
CNNDoublets::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CNNDoublets::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CNNDoublets::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNNDoublets::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNDoublets);
