// -*- C++ -*-
//
// Package:    CNNFiltering/CNN_TF_Test
// Class:      CNN_TF_Test
//
/**\class CNN_TF_Test CNN_TF_Test.cc CNNFiltering/CNN_TF_Test/plugins/CNN_TF_Test.cc

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

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class CNN_TF_Test : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit CNN_TF_Test(const edm::ParameterSet&);
  ~CNN_TF_Test();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  int particleBit();

  // ----------member data ---------------------------

  int doubletSize;
  std::string processName_;
  edm::EDGetTokenT<IntermediateHitDoublets> intHitDoublets_;
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
CNN_TF_Test::CNN_TF_Test(const edm::ParameterSet& iConfig):
processName_(iConfig.getParameter<std::string>("processName")),
intHitDoublets_(consumes<IntermediateHitDoublets>(iConfig.getParameter<edm::InputTag>("doublets"))),
tpMap_(consumes<ClusterTPAssociation>(iConfig.getParameter<edm::InputTag>("tpMap")))
{

  // usesResource("TFileService");
  //
  // edm::Service<TFileService> fs;
  // cnntree = fs->make<TTree>("CNNTree","Doublets Tree");

  // cnntree->Branch("test",      &test,          "test/I");

  edm::InputTag beamSpotTag = iConfig.getParameter<edm::InputTag>("beamSpot");
  bsSrc_ = consumes<reco::BeamSpot>(beamSpotTag);

  infoPileUp_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter< edm::InputTag >("infoPileUp"));

  padHalfSize = 8;
  padSize = (int)(padHalfSize*2);
  tParams = 26;

}


CNN_TF_Test::~CNN_TF_Test()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
CNN_TF_Test::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
  // std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

  // std::cout<<"CNNDoublets Analyzer"<<std::endl;

  edm::Handle<IntermediateHitDoublets> iHd;
  iEvent.getByToken(intHitDoublets_,iHd);

  edm::Handle<ClusterTPAssociation> tpClust;
  iEvent.getByToken(tpMap_,tpClust);

  // test = iEvent.id().event();
  //
  // cnntree->Fill();

  int eveNumber = iEvent.id().event();
  int runNumber = iEvent.id().run();
  int lumNumber = iEvent.id().luminosityBlock();

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

  std::vector< RecHitsSortedInPhi::Hit> hits;


  std::vector< std::vector< float>> hitPars;

  for (std::vector<IntermediateHitDoublets::LayerPairHitDoublets>::const_iterator lIt = iHd->layerSetsBegin(); lIt != iHd->layerSetsEnd(); ++lIt)
  {

    for (size_t i = 0; i < lIt->doublets().size(); i++)
    {

      hitPars.clear();

      hits.push_back(lIt->doublets().hit(i, HitDoublets::inner)); //TODO CHECK EMPLACEBACK
      hits.push_back(lIt->doublets().hit(i, HitDoublets::outer));

      float x = (hits[0]->hit()->globalState()).position.x();
      float y = (hits[0]->hit()->globalState()).position.y();
      float z = (hits[0]->hit()->globalState()).position.z();

      // Load graph
      tensorflow::setLogging("3");
      edm::FileInPath modelFilePath("/lustre/home/adrianodif/jpsiphi/MCs/QCDtoPhiML/CMSSW_10_2_1/tmp/test_graph_tfadd.pb");
      tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(modelFilePath.fullPath());
      tensorflow::Session* session = tensorflow::createSession(graphDef);

      tensorflow::Tensor inputX(tensorflow::DT_FLOAT, {});
      tensorflow::Tensor inputY(tensorflow::DT_FLOAT, {});


      inputX.scalar<float>()() = x;
      inputY.scalar<float>()() = y;

      std::vector<tensorflow::Tensor> outputs;
      tensorflow::run(session, { { "x_const", inputX }, { "y_const", inputY } },
                    { "x_y_sum" }, &outputs);

      std::cout << outputs[0].DebugString() << std::endl;
      std::cout << z << std::endl;


      }


    }


}


// ------------ method called once each job just before starting event loop  ------------
void
CNN_TF_Test::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
CNN_TF_Test::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CNN_TF_Test::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNN_TF_Test);
