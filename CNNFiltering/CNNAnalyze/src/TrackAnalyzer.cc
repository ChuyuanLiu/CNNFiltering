#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TLorentzVector.h"
#include "TTree.h"
#include <vector>
#include <sstream>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

//
// class declaration
//

class TrackAnalyzer : public edm::EDAnalyzer {
public:
  explicit TrackAnalyzer(const edm::ParameterSet&);
  ~TrackAnalyzer() override;

  bool isAncestor(const reco::Candidate* ancestor, const reco::Candidate * particle);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> TrakCollection_;

};

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig):
TrakCollection_(consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("PFCandidates")))
{

}

TrackAnalyzer::~TrackAnalyzer() {}

//
// member functions
//




// ------------ method called for each event  ------------
void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //  using namespace edm;
  using namespace edm;
  using namespace std;
  using namespace reco;
  typedef Candidate::LorentzVector LorentzVector;

  edm::Handle<edm::View<pat::PackedCandidate> > trak;
  iEvent.getByToken(TrakCollection_,trak);

  for (size_t i = 0; i < trak->size(); i++)
  {

    auto posTrack = trak->at(i);

    if(!(posTrack.hasTrackDetails())) continue;

    std::cout << posTrack.kaonId() << std::endl;
    // std::cout << posTrack.pionId() << std::endl;
    // std::cout << posTrack.muonId() << std::endl;
    // std::cout << posTrack.elecId() << std::endl;
    // std::cout << posTrack.elseId() << std::endl;

    //std::cout << theTrack->getTest() << std::endl;

  }

  }


// ------------ method called once each job just before starting event loop  ------------
void TrackAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TrackAnalyzer::endJob() {}

// ------------ method called when starting to processes a run  ------------
void TrackAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a run  ------------
void TrackAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void TrackAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void TrackAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(TrackAnalyzer);
