#include "../interface/TrackAnalyzer.h"

//Headers for the data items
#include <DataFormats/TrackReco/interface/TrackFwd.h>
#include <DataFormats/TrackReco/interface/Track.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/Common/interface/View.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include <DataFormats/PatCandidates/interface/Muon.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include "DataFormats/Math/interface/deltaR.h"

//Headers for services and tools
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"

#include "TMath.h"
#include "Math/VectorUtil.h"
#include "TVector3.h"

#include "TLorentzVector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig):
TrakCollection_(consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("PFCandidates")))
// muons_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
// muonPtCut_(iConfig.existsAs<double>("MuonPtCut") ? iConfig.getParameter<double>("MuonPtCut") : 0.7),
// thebeamspot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotTag"))),
// thePVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexTag"))),
// TriggerCollection_(consumes<std::vector<pat::TriggerObjectStandAlone>>(iConfig.getParameter<edm::InputTag>("TriggerInput"))),
// triggerResults_Label(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResults"))),
// dimuonSelection_(iConfig.existsAs<std::string>("dimuonSelection") ? iConfig.getParameter<std::string>("dimuonSelection") : ""),
// addCommonVertex_(iConfig.getParameter<bool>("addCommonVertex")),
// addMuonlessPrimaryVertex_(iConfig.getParameter<bool>("addMuonlessPrimaryVertex")),
// resolveAmbiguity_(iConfig.getParameter<bool>("resolvePileUpAmbiguity")),
// addMCTruth_(iConfig.getParameter<bool>("addMCTruth")),
// HLTFilters_(iConfig.getParameter<std::vector<std::string>>("HLTFilters"))
{


}


TrackAnalyzer::~TrackAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to produce the data  ------------

void
TrackAnalyzer::analyze(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
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

    auto theTrack = posTrack.bestTrack();

    std::cout << theTrack.getTest() << std::endl;

  }

}


//define this as a plug-in
DEFINE_FWK_MODULE(TrackAnalyzer);
