// -*- C++ -*-
//
// Package:    CNNTracks
// Class:      CNNTracks
//
// Description: Dimuon(mu+ mu-)  producer
//
// Author:  Adriano Di Florio
//    based on : Alberto Sanchez Hernandez Onia2MuMu code
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TLorentzVector.h"
#include "TTree.h"

#include <DataFormats/PatCandidates/interface/CompositeCandidate.h>
#include <DataFormats/PatCandidates/interface/Muon.h>
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <CommonTools/UtilAlgos/interface/StringCutObjectSelector.h>
#include "RecoVertex/VertexTools/interface/InvariantMassFromVertex.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

//
// class declaration
//

class CNNTracks:public edm::EDAnalyzer {
      public:
	explicit CNNTracks(const edm::ParameterSet &);
	~CNNTracks() override;

	static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      private:
        UInt_t getTriggerBits(const edm::Event &);
        bool   isAncestor(const reco::Candidate *, const reco::Candidate *);
        const  reco::Candidate* GetAncestor(const reco::Candidate *);

	void beginJob() override;
	void analyze(const edm::Event &, const edm::EventSetup &) override;
	void endJob() override;

	void beginRun(edm::Run const &, edm::EventSetup const &) override;
	void endRun(edm::Run const &, edm::EventSetup const &) override;
	void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
	void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

	// ----------member data ---------------------------
  edm::EDGetTokenT<edm::Association<reco::GenParticleCollection>> TrackGenMap_;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> TrakCollection_;
  edm::EDGetTokenT<edm::View<pat::Muon>> Muons_;
  edm::EDGetTokenT<reco::VertexCollection> ThePVs_;

	UInt_t    run;
	ULong64_t event;
  UInt_t    lumiblock;
  UInt_t    pu;

  Double_t pt, eta, phi, pdg, mompdg;
  Double_t charge, dxy, dz, NPixelHits, NStripHits, NTrackhits;
  Double_t NBPixHits, NPixLayers, NTraLayers, NStrLayers, NBPixLayers;

  std::array<double,20> hltword;
  std::array< std::array <double,8>, 25 > hitCoords;
  std::array< std::array <double,13>, 25 > pixelInfos;
  std::array< std::array <double,20>, 25 > pixelADC;
  std::array< std::array <double,20>, 25 > pixelADCx;
  std::array< std::array <double,20>, 25 > pixelADCy;
  std::array< std::array <double,7>, 25 > stripInfos;
  std::array< std::array <double,20>, 25 > stripADC;


	TTree *track_tree;



};

//
// constructors and destructor
//

CNNTracks::CNNTracks(const edm::ParameterSet & iConfig):
TrackGenMap_(consumes<edm::Association<reco::GenParticleCollection>>(iConfig.getParameter<edm::InputTag>("TrackMatcher"))),
TrakCollection_(consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
Muons_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
ThePVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexTag")))
{
  edm::Service < TFileService > fs;
  track_tree = fs->make < TTree > ("CnnTracks", "Tree of Tracks");

  track_tree->Branch("run",       &run,       "run/i");
  track_tree->Branch("event",     &event,     "event/l");
  track_tree->Branch("lumiblock", &lumiblock, "lumiblock/i");

  for (size_t i = 1; i < 65; i++)
  {
    track_tree->Branch(("hlt_word_" + std::to_string(i)).c_str(),    &hltword[i-1], ("hlt_word_" + std::to_string(i) + "/D").c_str());
  }

  track_tree->Branch("pu",    &pu,    "pu/i");

  track_tree->Branch("pt",     &pt,     "pt/D");
  track_tree->Branch("eta",    &eta,    "eta/D");
  track_tree->Branch("phi",    &phi,    "phi/D");
  track_tree->Branch("charge", &charge, "charge/D");
  track_tree->Branch("dxy",    &dxy,    "dxy/D");
  track_tree->Branch("dz",     &dz,     "dz/D");

  track_tree->Branch("pdg",     &pdg,     "pdg/D");
  track_tree->Branch("mompdg",  &mompdg,  "mompdg/D");

  track_tree->Branch("NPixelHits",  &NPixelHits,  "NPixelHits/D");
  track_tree->Branch("NStripHits",  &NStripHits,  "NStripHits/D");
  track_tree->Branch("NTrackhits",  &NTrackhits,  "NTrackhits/D");
  track_tree->Branch("NBPixHits",   &NBPixHits,   "NBPixHits/D");
  track_tree->Branch("NPixLayers",  &NPixLayers,  "NPixLayers/D");
  track_tree->Branch("NTraLayers",  &NTraLayers,  "NTraLayers/D");
  track_tree->Branch("NStrLayers",  &NStrLayers,  "NStrLayers/D");
  track_tree->Branch("NBPixLayers", &NBPixLayers, "NBPixLayers/D");

  for(int j = 0; j < 25; j++)
  {

    track_tree->Branch("n_" + std::to_string(j),   &hitCoords[j][0], "n_" + std::to_string(j) + "/D");
    track_tree->Branch("x_" + std::to_string(j),   &hitCoords[j][1], "x_" + std::to_string(j) + "/D");
    track_tree->Branch("y_" + std::to_string(j),   &hitCoords[j][2], "y_" + std::to_string(j) + "/D");
    track_tree->Branch("z_" + std::to_string(j),   &hitCoords[j][3], "z_" + std::to_string(j) + "/D");
    track_tree->Branch("phi_" + std::to_string(j), &hitCoords[j][4], "phi_" + std::to_string(j) + "/D");
    track_tree->Branch("r_" + std::to_string(j),   &hitCoords[j][5], "r_" + std::to_string(j) + "/D");
    track_tree->Branch("ax1_" + std::to_string(j), &hitCoords[j][6], "ax1_" + std::to_string(j) + "/D");
    track_tree->Branch("ax2_" + std::to_string(j), &hitCoords[j][7], "ax2_" + std::to_string(j) + "/D");

    track_tree->Branch("pix_n_" + std::to_string(j),      &pixelInfos[j][0], "pix_n_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_x_" + std::to_string(j),      &pixelInfos[j][1], "pix_x_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_y_" + std::to_string(j),      &pixelInfos[j][2], "pix_y_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_size_" + std::to_string(j),   &pixelInfos[j][3], "pix_size_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_sizeX_" + std::to_string(j),  &pixelInfos[j][4], "pix_sizeX_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_sizeY_" + std::to_string(j),  &pixelInfos[j][5], "pix_sizeY_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_charge_" + std::to_string(j), &pixelInfos[j][6], "pix_charge_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_ovfX_" + std::to_string(j),   &pixelInfos[j][7], "pix_ovfX_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_ovfY_" + std::to_string(j),   &pixelInfos[j][8], "pix_ovfY_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_skew_" + std::to_string(j),   &pixelInfos[j][9], "pix_skew_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_big_" + std::to_string(j),    &pixelInfos[j][10], "pix_big_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_bad_" + std::to_string(j),    &pixelInfos[j][11], "pix_bad_" + std::to_string(j) + "/D");
    track_tree->Branch("pix_edg_" + std::to_string(j),    &pixelInfos[j][12], "pix_edg_" + std::to_string(j) + "/D");

    track_tree->Branch("strip_n_" + std::to_string(j),        &stripInfos[j][0], "strip_n_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_dim_" + std::to_string(j),      &stripInfos[j][1], "strip_dim_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_charge_" + std::to_string(j),   &stripInfos[j][2], "strip_charge_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_baryc_" + std::to_string(j),    &stripInfos[j][3], "strip_baryc_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_first_" + std::to_string(j),    &stripInfos[j][4], "strip_first_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_merge_" + std::to_string(j),    &stripInfos[j][5], "strip_merge_" + std::to_string(j) + "/D");
    track_tree->Branch("strip_size_" + std::to_string(j),     &stripInfos[j][6], "strip_size_" + std::to_string(j) + "/D");

    for (size_t i = 0; i < 20; i++)
    {
      track_tree->Branch("pix_adc_" + std::to_string(j) + "_" + std::to_string(i),        &pixelADC[j][i], "pix_adc_" + std::to_string(j) + "_" + std::to_string(i) + "/D");
    }

    for (size_t i = 0; i < 20; i++)
    {
      track_tree->Branch("pix_adcX_" + std::to_string(j) + "_" + std::to_string(i),        &pixelADCX[j][i], "pix_adcX_" + std::to_string(j) + "_" + std::to_string(i) + "/D");
    }

    for (size_t i = 0; i < 20; i++)
    {
      track_tree->Branch("pix_adcY_" + std::to_string(j) + "_" + std::to_string(i),        &pixelADCY[j][i], "pix_adcY_" + std::to_string(j) + "_" + std::to_string(i) + "/D");
    }

    for (size_t i = 0; i < 20; i++)
    {
      track_tree->Branch("strip_adc_" + std::to_string(j) + "_" + std::to_string(i),        &stripADCY[j][i], "strip_adc_" + std::to_string(j) + "_" + std::to_string(i) + "/D");
    }





  }



}

CNNTracks::~CNNTracks() {}

//
// member functions
//


/* Grab Trigger information. Save it in variable trigger, trigger is an uint between 0 and 256, in binary it is:
   (pass 2)(pass 1)(pass 0)
   ex. 7 = pass 0, 1 and 2
   ex. 6 = pass 1, 2
   ex. 1 = pass 0
*/


UInt_t CNNTracks::getTriggerBits(const edm::Event& iEvent, int triggerChunk) {

  UInt_t trigger = 0;

  edm::Handle< edm::TriggerResults > triggerResults_handle;
  iEvent.getByToken( triggerResults_Label , triggerResults_handle);

  if (triggerResults_handle.isValid()) {
     const edm::TriggerNames & TheTriggerNames = iEvent.triggerNames(*triggerResults_handle);
     unsigned int NTRIGGERS = HLTs_.size();

     int lastTrig = triggerChunk*12;
     if(lastTrig > NTRIGGERS) continue;

     for (unsigned int i = 0; i < lastTrig; i++) {
       if(trigger)
        for (int version = 1; version < 20; version++) {
           std::stringstream ss;
           ss << HLTs_[i] << "_v" << version;
           unsigned int bit = TheTriggerNames.triggerIndex(edm::InputTag(ss.str()).label());
           if (bit < triggerResults_handle->size() && triggerResults_handle->accept(bit) && !triggerResults_handle->error(bit)) {
              trigger += (1<<i);
              break;
           }
        }
     }
   } else std::cout << "*** NO triggerResults found " << iEvent.id().run() << "," << iEvent.id().event() << std::endl;

   return trigger;
}

// ------------ method called for each event  ------------
void CNNTracks::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {

  edm::Handle<reco::VertexCollection> primaryVertices_handle;
  iEvent.getByToken(ThePVs_, primaryVertices_handle);

  edm::Handle<edm::View<pat::PackedCandidate> > track;
  iEvent.getByToken(TrakCollection_,track);

  edm::Handle<edm::View<pat::Muon> > muons;
  iEvent.getByToken(Muons_,muons);

  run       = iEvent.id().run();
  event     = iEvent.id().event();
  lumiblock = iEvent.id().luminosityBlock();

  pu = 0;
  if (primaryVertices_handle.isValid()) pu = (int) primaryVertices_handle->size();

  for (size_t i = 0; i < 64; i++)
  {
    hltword[i] = getTriggerBits(iEvent,i+1);
  }

  bool atLeastOne = false;

  for (size_t i = 0; i < track->size(); i++) {

    auto t = track->at(i);
    if(!(t.trackHighPurity())) continue;
    if(!(t.hasTrackDetails())) continue;

    auto refTrack = track->refAt(i);
    int pdgId = -9999, momPdgId = -9999;

    if(theGenMap->contains(refTrack.id()))
    {
      if(((*theGenMap)[edm::Ref<edm::View<pat::PackedCandidate>>(track, i)]).isNonnull())
      {
        auto genParticle = ((*theGenMap)[edm::Ref<edm::View<pat::PackedCandidate>>(track, i)]);
        // auto genParticle = dynamic_cast <const reco::GenParticle *>(((*theGenMap)[edm::Ref<edm::View<pat::PackedCandidate>>(track, i)]));
        pdgId = genParticle->pdgId();
        if(genParticle->numberOfMothers()>0)
        if(genParticle->motherRef().isNonnull())
        momPdgId = genParticle->motherRef()->pdgId();
      }
    }

    int noHits = t.hitCoords_.size();
    int maxHits = 25;
    int minHits = -std::max(maxHits,noHits);

    pt  = t.pt();
    eta = t.eta();
    phi = t.phi();
    charge = t.charge();
    dxy = t.bestTrack()->dxy();
    dz  = t.bestTrack()->dz();
    NPixelHits = t.bestTrack()->hitPattern().numberOfValidPixelHits();
    NStripHits = t.bestTrack()->hitPattern().numberOfValidStripHits();
    NTrackhits = t.bestTrack()->hitPattern().numberOfValidTrackerHits();
    NBPixHits  = t.bestTrack()->hitPattern().numberOfValidStripHits();
    NPixLayers = t.bestTrack()->hitPattern().pixelLayersWithMeasurement();
    NTraLayers = t.bestTrack()->hitPattern().trackerLayersWithMeasurement();
    NStrLayers = t.bestTrack()->hitPattern().stripLayersWithMeasurement();
    NBPixLayers = t.bestTrack()->hitPattern().pixelBarrelLayersWithMeasurement();

    pdg = pdgId;
    mompdg = momPdgId;
    for(int j = 0; j<minHits;j++)
    {

      auto coords  = t.hitCoords_[j];
      auto pixinf  = t.pixelInfos_[j];
      auto pixadc  = t.pixelADC_[j];
      auto pixadx  = t.pixelADCx_[j];
      auto pixady  = t.pixelADCy_[j];
      auto strinf  = t.stripInfos_[j];
      auto stradc  = t.stripADC_[j];

      for (size_t i = 0; i < 8; i++)
      {
        hitCoords[j][i] = coords[i];
      }
      for (size_t i = 0; i < 13; i++)
      {
        pixelInfos[j][i] = pixinf[i];
      }
      for (size_t i = 0; i < 20; i++)
      {
        pixelADC[j][i] = pixadc[i];
        pixelADCx[j][i] = pixadx[i];
        pixelADCy[j][i] = pixady[i];
        stripADC[j][i] = stradc[i];
      }

      for (size_t i = 0; i < 7; i++)
      {
        stripInfos[j][i] = strinf[i];
      }

    }

  }
  if(atLeastOne)
    track_tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void CNNTracks::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CNNTracks::endJob() {}

// ------------ method called when starting to processes a run  ------------
void CNNTracks::beginRun(edm::Run const &, edm::EventSetup const &) {}

// ------------ method called when ending the processing of a run  ------------
void CNNTracks::endRun(edm::Run const &, edm::EventSetup const &) {}

// ------------ method called when starting to processes a luminosity block  ------------
void CNNTracks::beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void CNNTracks::endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CNNTracks::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
	//The following says we do not know what parameters are allowed so do no validation
	// Please change this to state exactly what you do use, even if it is no parameters
	edm::ParameterSetDescription desc;
	desc.setUnknown();
	descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CNNTracks);
