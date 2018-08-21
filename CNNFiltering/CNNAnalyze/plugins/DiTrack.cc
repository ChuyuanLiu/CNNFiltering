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
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
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
#include "RecoVertex/VertexTools/interface/InvariantMassFromVertex.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TLorentzVector.h"
#include "TTree.h"

//
// class declaration
//

class DiTrack:public edm::EDAnalyzer {
      public:
	explicit DiTrack(const edm::ParameterSet &);
	~DiTrack() override;

	static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      private:
        UInt_t getTriggerBits(const edm::Event& iEvent, const edm::Handle< edm::TriggerResults >& triggerResults_handle);
        bool   isAncestor(const reco::Candidate *, const reco::Candidate *);
        const  reco::Candidate* GetAncestor(const reco::Candidate *);
        UInt_t isTriggerMatched(const pat::CompositeCandidate *diTrig_Candidate);

	void beginJob() override;
	void analyze(const edm::Event &, const edm::EventSetup &) override;
	void endJob() override;

	void beginRun(edm::Run const &, edm::EventSetup const &) override;
	void endRun(edm::Run const &, edm::EventSetup const &) override;
	void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
	void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

	// ----------member data ---------------------------
	std::string file_name;
	// edm::EDGetTokenT<pat::CompositeCandidateCollection> diTrak_label;
  edm::EDGetTokenT<edm::View<reco::Track>> alltracks_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_Label;
  std::vector<double> ditrakMassCuts_;
  std::vector<double> MassTraks_;
	bool isMC_;
  bool OnlyBest_;
  std::vector<std::string>  HLTs_;
  std::vector<std::string>  HLTFilters_;

  reco::Candidate::LorentzVector convertVector(const math::XYZTLorentzVectorF& v);
  bool IsTheSame(const pat::PackedCandidate& t1, const pat::PackedCandidate& t2);

  const pat::CompositeCandidate makeTTTriggerCandidate(const pat::TriggerObjectStandAlone& t1,
						    const pat::TriggerObjectStandAlone& t2);
  const pat::CompositeCandidate makeTTCandidate(
                                            const reco::Track& trakP,
                                            const reco::Track& trakN
                                          );

  bool MatchByDRDPt(const pat::PackedCandidate t1, const pat::TriggerObjectStandAlone t2);
  float DeltaR(const pat::PackedCandidate t1, const pat::TriggerObjectStandAlone t2);

  int candidates;
  int nevents;
  int ndimuon;
  int nreco;
  float maxDeltaR;
  float maxDPtRel;

	UInt_t    run;
	ULong64_t event;
  UInt_t    lumiblock;

  UInt_t    trigger;
  UInt_t    tMatchOne,tMatchTwo;

  UInt_t charge;
  UInt_t negPixHits, posPixHits;

  Double_t ditrak_m, ditrak_p, ditrak_pt, ditrak_eta, ditrak_phi,ditrak_vProb;

	TLorentzVector ditrak_p4;
	TLorentzVector trakP_p4;
	TLorentzVector trakN_p4;

  TLorentzVector ditrig_p4;
  TLorentzVector trigP_p4;
  TLorentzVector trigN_p4;

	UInt_t numPrimaryVertices;

	TTree *ditrak_tree;

  UInt_t nditrak, ntraks;
  UInt_t tJ, tI;

};


const pat::CompositeCandidate DiTrack::makeTTCandidate(
                                          const reco::Track& trakP,
                                          const reco::Track& trakN
                                         ){

  pat::CompositeCandidate TTCand;
  // TTCand.addDaughter(trakP,"trakP");
  // TTCand.addDaughter(trakN,"trakN");
  TTCand.setCharge(trakP.charge()+trakN.charge());

  double m_kaon1 = MassTraks_[0];
  math::XYZVector mom_kaon1 = trakP.momentum();
  double e_kaon1 = sqrt(m_kaon1*m_kaon1 + mom_kaon1.Mag2());
  math::XYZTLorentzVector p4_kaon1 = math::XYZTLorentzVector(mom_kaon1.X(),mom_kaon1.Y(),mom_kaon1.Z(),e_kaon1);
  double m_kaon2 = MassTraks_[1];
  math::XYZVector mom_kaon2 = trakN.momentum();
  double e_kaon2 = sqrt(m_kaon2*m_kaon2 + mom_kaon2.Mag2());
  math::XYZTLorentzVector p4_kaon2 = math::XYZTLorentzVector(mom_kaon2.X(),mom_kaon2.Y(),mom_kaon2.Z(),e_kaon2);
  reco::Candidate::LorentzVector vTT = p4_kaon1 + p4_kaon2;
  TTCand.setP4(vTT);

  return TTCand;
}

//
// constructors and destructor
//

DiTrack::DiTrack(const edm::ParameterSet & iConfig):
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
//triggerResults_Label(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResults"))),
ditrakMassCuts_(iConfig.getParameter<std::vector<double>>("TrakTrakMassCuts")),
MassTraks_(iConfig.getParameter<std::vector<double>>("MassTraks"))
//HLTs_(iConfig.getParameter<std::vector<std::string>>("HLTs")),
//HLTFilters_(iConfig.getParameter<std::vector<std::string>>("Filters"))
{
  edm::Service < TFileService > fs;
  ditrak_tree = fs->make < TTree > ("DiTrakDiTrigTree", "Tree of ditrakditrig");

  ditrak_tree->Branch("run",      &run,      "run/i");
  ditrak_tree->Branch("event",    &event,    "event/l");
  ditrak_tree->Branch("lumiblock",&lumiblock,"lumiblock/i");

  ditrak_tree->Branch("tI",&tI,"tI/i");
  ditrak_tree->Branch("tJ",&tJ,"tJ/i");

  ditrak_tree->Branch("posPixHits",&posPixHits,"posPixHits/i");
  ditrak_tree->Branch("negPixHits",&negPixHits,"negPixHits/i");

  // ditrak_tree->Branch("nditrak",    &nditrak,    "nditrak/i");
  // ditrak_tree->Branch("ntraks",   &ntraks,   "ntraks/i");
  // ditrak_tree->Branch("trigger",  &trigger,  "trigger/i");
  ditrak_tree->Branch("charge",   &charge,   "charge/I");

  ditrak_tree->Branch("ditrak_m",   "TLorentzVector", &ditrak_m);
  ditrak_tree->Branch("ditrak_p",   "TLorentzVector", &ditrak_p);
  ditrak_tree->Branch("ditrak_pt",  "TLorentzVector", &ditrak_pt);
  ditrak_tree->Branch("ditrak_eta", "TLorentzVector", &ditrak_eta);
  ditrak_tree->Branch("ditrak_phi", "TLorentzVector", &ditrak_phi);
  ditrak_tree->Branch("ditrak_vProb", "TLorentzVector", &ditrak_vProb);

  // ditrak_tree->Branch("numPrimaryVertices", &numPrimaryVertices, "numPrimaryVertices/i");

  candidates = 0;
  nevents = 0;
  ndimuon = 0;
  nreco = 0;
  maxDeltaR = 0.01;
  maxDPtRel = 2.0;

}

DiTrack::~DiTrack() {}

/* Grab Trigger information. Save it in variable trigger, trigger is an uint between 0 and 256, in binary it is:
   (pass 2)(pass 1)(pass 0)
   ex. 7 = pass 0, 1 and 2
   ex. 6 = pass 1, 2
   ex. 1 = pass 0
*/

UInt_t DiTrack::getTriggerBits(const edm::Event& iEvent, const edm::Handle< edm::TriggerResults >& triggerResults_handle) {

  UInt_t trigger = 0;
  const edm::TriggerNames & names = iEvent.triggerNames( *triggerResults_handle );

     unsigned int NTRIGGERS = HLTs_.size();

     for (unsigned int i = 0; i < NTRIGGERS; i++) {
        for (int version = 1; version < 20; version++) {
           std::stringstream ss;
           ss << HLTs_[i] << "_v" << version;
           unsigned int bit = names.triggerIndex(edm::InputTag(ss.str()).label());
           if (bit < triggerResults_handle->size() && triggerResults_handle->accept(bit) && !triggerResults_handle->error(bit)) {
              trigger += (1<<i);
              break;
           }
        }
     }

   return trigger;
}

// ------------ method called for each event  ------------
void DiTrack::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {

  using namespace edm;
  using namespace std;
  using namespace reco;
  // edm::Handle<pat::CompositeCandidateCollection> ditraks;
  // iEvent.getByToken(diTrak_label,ditraks);

  // edm::Handle<pat::CompositeCandidateCollection> ditrigs;
  // iEvent.getByToken(diTrig_label,ditrigs);

  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality= reco::TrackBase::qualityByName(theTrackQuality);

  edm::Handle<edm::View<reco::Track> >  trackCollection;
  iEvent.getByToken(alltracks_, trackCollection);

  edm::ESHandle<TransientTrackBuilder> theTTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTTBuilder);
  KalmanVertexFitter vtxFitter(true);


  run       = iEvent.id().run();
  event     = iEvent.id().event();
  lumiblock = iEvent.id().luminosityBlock();

  numPrimaryVertices = 0;
  // if (primaryVertices_handle.isValid()) numPrimaryVertices = (int) primaryVertices_handle->size();
  //
  // edm::Handle< edm::TriggerResults > triggerResults_handle;
  // iEvent.getByToken( triggerResults_Label , triggerResults_handle);

  trigger = 0;
  //
  // if (triggerResults_handle.isValid())
  //   trigger = getTriggerBits(iEvent,triggerResults_handle);
  // else std::cout << "*** NO triggerResults found " << iEvent.id().run() << "," << iEvent.id().event() << std::endl;

  nditrak  = 0;
  ntraks = 0;

  float TrakTrakMassMax_ = ditrakMassCuts_[1];
  float TrakTrakMassMin_ = ditrakMassCuts_[0];


  // for (std::vector<pat::PackedCandidate>::const_iterator posTrack = filteredTracks.begin(), trakend=filteredTracks.end(); posTrack!= trakend; ++posTrack)

  for(edm::View<reco::Track>::size_type i=0; i<trackCollection->size(); ++i)
  {
           auto posTrack = trackCollection->refAt(i);
           bool trkQual  = posTrack->quality(trackQuality);
           auto hitPattern = posTrack->hitPattern();

           posPixHits = hitPattern.numberOfValidPixelHits();
           // std::cout << "- No Pixel Hits :" << pixHits << std::endl;
           if(posPixHits < 4)
             continue;

           tI = (UInt_t)(i);

           if(!trkQual)
             continue;

           if(posTrack->charge() <= 0 ) continue;
           if(posTrack->pt()<0.9) continue;
	std::cout<<"postrack"<< std::endl;
     for(edm::View<reco::Track>::size_type j=0; j<trackCollection->size(); ++j)
     {

       ditrak_m     = 0.0;
       ditrak_p     = 0.0;
       ditrak_pt    = 0.0;
       ditrak_eta   = 0.0;
       ditrak_phi   = 0.0;
       ditrak_vProb = 0.0;

       auto negTrack = trackCollection->refAt(j);

       negPixHits = negTrack->hitPattern().numberOfValidPixelHits();
       // std::cout << "- No Pixel Hits :" << pixHits << std::endl;
       if(negPixHits < 4)
         continue;

       tJ = (UInt_t)(j);

       if(!(negTrack->quality(trackQuality)))
         continue;

       if(negTrack->charge() <= 0 ) continue;
       if(negTrack->pt()<0.9) continue;
	std::cout<<"postrack"<< std::endl;
       pat::CompositeCandidate TTCand = makeTTCandidate(*posTrack,*negTrack);

       if ( !(TTCand.mass() < TrakTrakMassMax_ && TTCand.mass() > TrakTrakMassMin_) )
        continue;
	std::cout<<"cand"<< std::endl;
       std::vector<TransientTrack> tt_ttks;
       tt_ttks.push_back(theTTBuilder->build(*negTrack));  // pass the reco::Track, not  the reco::TrackRef (which can be transient)
       tt_ttks.push_back(theTTBuilder->build(*posTrack));

       TransientVertex ttVertex = vtxFitter.vertex(tt_ttks);
       CachingVertex<5> VtxForInvMass = vtxFitter.vertex( tt_ttks );
	std::cout<<"fit"<< std::endl;
       double vChi2 = ttVertex.totalChiSquared();
       double vNDF  = ttVertex.degreesOfFreedom();
       ditrak_vProb = TMath::Prob(vChi2,(int)vNDF);

       ditrak_m     = TTCand.mass();
       ditrak_p     = TTCand.pt();
       ditrak_pt    = TTCand.p();
       ditrak_eta   = TTCand.eta();
       ditrak_phi   = TTCand.phi();

       if(ditrak_vProb>0.0)
        ditrak_tree->Fill();

	std::cout<<"fill"<< std::endl;
           } // loop over second track
         }   // loop on track candidates
}

// ------------ method called once each job just before starting event loop  ------------
void DiTrack::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DiTrack::endJob() {}

// ------------ method called when starting to processes a run  ------------
void DiTrack::beginRun(edm::Run const &, edm::EventSetup const &) {}

// ------------ method called when ending the processing of a run  ------------
void DiTrack::endRun(edm::Run const &, edm::EventSetup const &) {}

// ------------ method called when starting to processes a luminosity block  ------------
void DiTrack::beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

// ------------ method called when ending the processing of a luminosity block  ------------
void DiTrack::endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DiTrack::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
	//The following says we do not know what parameters are allowed so do no validation
	// Please change this to state exactly what you do use, even if it is no parameters
	edm::ParameterSetDescription desc;
	desc.setUnknown();
	descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DiTrack);
