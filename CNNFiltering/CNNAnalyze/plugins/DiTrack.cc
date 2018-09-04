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

#include "TH2F.h"
#include "TTree.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TLorentzVector.h"
#include "TTree.h"

#include <iostream>
#include <string>
#include <fstream>

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
  int seqNumber_;
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

  float padHalfSize;
  int padSize, tParams;

	UInt_t    run;
	ULong64_t event;
  UInt_t    lumiblock;

  UInt_t    trigger;
  UInt_t    tMatchOne,tMatchTwo;

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

  float pt, eta, phi, p, chi2n, d0, dx, dz, sharedFraction,sharedMomFraction;
  int nhit, nhpxf, nhtib, nhtob, nhtid, nhtec, nhpxb, nHits, trackPdg,trackMomPdg;
  int eveNumber, runNumber, lumNumber;

  std::vector<float>  x, y, z, phi_hit, r, c_x, c_y, charge, ovfx, ovfy;
  std::vector<float> ratio, pdgId, motherPdgId, size, sizex, sizey;
  //std::vector<TH2> hitClust;

  std::vector<float> hitPixel0, hitPixel1, hitPixel2, hitPixel3, hitPixel4;
  std::vector<float> hitPixel5, hitPixel6, hitPixel7, hitPixel8, hitPixel9;

  std::vector< std::vector<float> > hitPixels;

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
seqNumber_(iConfig.getParameter<int>("seqNumber")),
alltracks_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("Tracks"))),
triggerResults_Label(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("Trigger"))),
ditrakMassCuts_(iConfig.getParameter<std::vector<double>>("TrakTrakMassCuts")),
MassTraks_(iConfig.getParameter<std::vector<double>>("MassTraks")),
HLTs_(iConfig.getParameter<std::vector<std::string>>("HLTs"))
//HLTFilters_(iConfig.getParameter<std::vector<std::string>>("Filters"))
{
  // edm::Service < TFileService > fs;
  // ditrak_tree = fs->make < TTree > ("DiTrakDiTrigTree", "Tree of ditrakditrig");
  //
  // ditrak_tree->Branch("run",      &run,      "run/i");
  // ditrak_tree->Branch("event",    &event,    "event/l");
  // ditrak_tree->Branch("lumiblock",&lumiblock,"lumiblock/i");
  //
  // ditrak_tree->Branch("tI",&tI,"tI/i");
  // ditrak_tree->Branch("tJ",&tJ,"tJ/i");
  //
  // ditrak_tree->Branch("posPixHits",&posPixHits,"posPixHits/i");
  // ditrak_tree->Branch("negPixHits",&negPixHits,"negPixHits/i");
  //
  // // ditrak_tree->Branch("nditrak",    &nditrak,    "nditrak/i");
  // // ditrak_tree->Branch("ntraks",   &ntraks,   "ntraks/i");
  // // ditrak_tree->Branch("trigger",  &trigger,  "trigger/i");
  // ditrak_tree->Branch("charge",   &charge,   "charge/I");
  //
  // ditrak_tree->Branch("ditrak_m",   "TLorentzVector", &ditrak_m);
  // ditrak_tree->Branch("ditrak_p",   "TLorentzVector", &ditrak_p);
  // ditrak_tree->Branch("ditrak_pt",  "TLorentzVector", &ditrak_pt);
  // ditrak_tree->Branch("ditrak_eta", "TLorentzVector", &ditrak_eta);
  // ditrak_tree->Branch("ditrak_phi", "TLorentzVector", &ditrak_phi);
  // ditrak_tree->Branch("ditrak_vProb", "TLorentzVector", &ditrak_vProb);

  // ditrak_tree->Branch("numPrimaryVertices", &numPrimaryVertices, "numPrimaryVertices/i");

  candidates = 0;
  nevents = 0;
  nditrak = 0;
  nreco = 0;
  maxDeltaR = 0.01;
  maxDPtRel = 2.0;

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

  for(int i = 0; i<10;i++)
    for(int j =0;j<padSize*padSize;j++)
      hitPixels[i].push_back(0.0);

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

  std::string fileName = "phicandsCNN.txt"// + std::to_string(lumiblock) + "_" ;
  //fileName = fileName + std::to_string(run) + "_" ;
  //fileName = fileName + std::to_string(event) + "_" ;
  //fileName = fileName + std::to_string(seqNumber_);
  //fileName = fileName + ".txt";

  std::ofstream outPhiFile(fileName, std::ofstream::app);

  // fileName = "generalTracksCNN_" + std::to_string(lumNumber) + "_" ;
  // fileName = fileName + std::to_string(runNumber) + "_" ;
  // fileName = fileName + std::to_string(eveNumber) + "_" ;
  // fileName = fileName + std::to_string(seqNumber_);
  // fileName = fileName + ".txt";
  // //std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
  // //fileName += "_" + processName_ + "_dnn_doublets.txt";
  // std::ofstream outPhiFile(fileName, std::ofstream::app);

  numPrimaryVertices = 0;
  // if (primaryVertices_handle.isValid()) numPrimaryVertices = (int) primaryVertices_handle->size();
  //
  edm::Handle< edm::TriggerResults > triggerResults_handle;
  iEvent.getByToken( triggerResults_Label , triggerResults_handle);

  trigger = 0;
  //
  if (triggerResults_handle.isValid())
    trigger = getTriggerBits(iEvent,triggerResults_handle);
  else std::cout << "*** NO triggerResults found " << iEvent.id().run() << "," << iEvent.id().event() << std::endl;

  nditrak  = 0;
  ntraks = 0;

  float TrakTrakMassMax_ = ditrakMassCuts_[1];
  float TrakTrakMassMin_ = ditrakMassCuts_[0];


  // for (std::vector<pat::PackedCandidate>::const_iterator posTrack = filteredTracks.begin(), trakend=filteredTracks.end(); posTrack!= trakend; ++posTrack)

  for(edm::View<reco::Track>::size_type k=0; k<trackCollection->size(); ++k)
  {
           auto posTrack = trackCollection->refAt(k);
           bool trkQual  = posTrack->quality(trackQuality);
           auto hitPattern = posTrack->hitPattern();

           posPixHits = hitPattern.numberOfValidPixelHits();
           // std::cout << "- No Pixel Hits :" << pixHits << std::endl;
           if(posPixHits < 3)
             continue;

           tI = (UInt_t)(k);

           if(!trkQual)
             continue;

           if(posTrack->charge() <= 0 ) continue;
           if(posTrack->pt()<0.7) continue;
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
       if(negPixHits < 3)
         continue;

       tJ = (UInt_t)(j);

       if(!(negTrack->quality(trackQuality)))
         continue;

       if(negTrack->charge() <= 0 ) continue;
       if(negTrack->pt()<0.7) continue;
       pat::CompositeCandidate TTCand = makeTTCandidate(*posTrack,*negTrack);

       if ( !(TTCand.mass() < TrakTrakMassMax_ && TTCand.mass() > TrakTrakMassMin_) )
        continue;
       std::vector<TransientTrack> tt_ttks;
       tt_ttks.push_back(theTTBuilder->build(*negTrack));  // pass the reco::Track, not  the reco::TrackRef (which can be transient)
       tt_ttks.push_back(theTTBuilder->build(*posTrack));

       TransientVertex ttVertex = vtxFitter.vertex(tt_ttks);
       CachingVertex<5> VtxForInvMass = vtxFitter.vertex( tt_ttks );
       double vChi2 = ttVertex.totalChiSquared();
       double vNDF  = ttVertex.degreesOfFreedom();
       ditrak_vProb = TMath::Prob(vChi2,(int)vNDF);

       ditrak_m     = TTCand.mass();
       ditrak_p     = TTCand.pt();
       ditrak_pt    = TTCand.p();
       ditrak_eta   = TTCand.eta();
       ditrak_phi   = TTCand.phi();

       if(ditrak_vProb>0.05)
       {

          outPhiFile << (float)run << "\t";
          outPhiFile << (float)event << "\t";
          outPhiFile << (float)lumiblock << "\t";
          outPhiFile << (float)(seqNumber_) << "\t";

          outPhiFile << (float)tI << "\t";
          outPhiFile << (float)tJ << "\t";

          outPhiFile << (float)posPixHits << "\t";
          outPhiFile << (float)negPixHits << "\t";

          outPhiFile << (float)ditrak_m << "\t";
          outPhiFile << (float)ditrak_p << "\t";
          outPhiFile << (float)ditrak_pt << "\t";
          outPhiFile << (float)ditrak_eta << "\t";
          outPhiFile << (float)ditrak_phi << "\t";
          outPhiFile << (float)ditrak_vProb << "\t";
          //outPhiFile << 542.1369 << std::endl;
          ++nditrak;

          UInt_t tIds[2] = {tI,tJ};

          for(int ii =0;ii<2;ii++)
          {
            UInt_t tt = tIds[ii];
            std::vector<double> theData;
            // std::cout << "Track ------------------- "<< std::endl;
            // std::cout << std::endl;
            std::map<int,const TrackerSingleRecHit*> theHits;
            std::map<int,bool> isBad,isEdge,isBig;
            std::map<int,double> hitSize,pdgIds,flagHit;

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
            if(pixHits < 4)
              continue;

            // pt    = (double)track->pt();
            // eta   = (double)track->eta();
            // phi   = (double)track->phi();
            // p     = (double)track->p();
            // chi2n = (double)track->normalizedChi2();
            // nhit  = (double)track->numberOfValidHits();
            // d0    = (double)track->d0();
            // dz    = (double)track->dz();

            // nhpxb   = hitPattern.numberOfValidPixelBarrelHits();
            // nhpxf   = hitPattern.numberOfValidPixelEndcapHits();
            // nhtib   = hitPattern.numberOfValidStripTIBHits();
            // nhtob   = hitPattern.numberOfValidStripTOBHits();
            // nhtid   = hitPattern.numberOfValidStripTIDHits();
            // nhtec   = hitPattern.numberOfValidStripTECHits();

            // std::cout<<nhit<<std::endl;
            theData.clear();

            theData.push_back((double)track->pt());
            theData.push_back((double)track->eta());
            theData.push_back((double)track->phi());
            theData.push_back((double)track->p());
            theData.push_back((double)track->normalizedChi2());
            theData.push_back((double)track->numberOfValidHits());
            theData.push_back((double)track->d0());
            theData.push_back((double)track->dz());

            theData.push_back((double)hitPattern.numberOfValidPixelBarrelHits());
            theData.push_back((double)hitPattern.numberOfValidPixelEndcapHits());
            theData.push_back((double)hitPattern.numberOfValidStripTIBHits());
            theData.push_back((double)hitPattern.numberOfValidStripTOBHits());
            theData.push_back((double)hitPattern.numberOfValidStripTIDHits());
            theData.push_back((double)hitPattern.numberOfValidStripTECHits());

            // std::cout << "track" << std::endl;
            for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
            {
              TrackerSingleRecHit const * hit= dynamic_cast<TrackerSingleRecHit const *>(*recHit);

              if(!hit)
                continue;
              if(!hit->hasPositionAndError())
                continue;

              DetId detId = (*recHit)->geographicalId();
              unsigned int subdetid = detId.subdetId();

              if(detId.det() != DetId::Tracker) continue;
              if (!((subdetid==1) || (subdetid==2))) continue;

              const SiPixelRecHit* pixHit = dynamic_cast<SiPixelRecHit const *>(hit);

              int hitLayer = -1;

              if (!pixHit)
                continue;

              if(subdetid==1) //barrel
                hitLayer = PXBDetId(detId).layer();
              else
              if(subdetid==2)
              {
                //int side = PXFDetId(detId).side();
                float z = (pixHit->globalState()).position.z();
                if(fabs(z)>28.0) hitLayer = 4;
                if(fabs(z)>36.0) hitLayer = 5;
                if(fabs(z)>44.0) hitLayer = 6;

                if(z<=0.0) hitLayer +=3;
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
                  //   if(!(thisBad || thisEdge || thisBig))
                  //     keepThis = true;
                  // if(isBad[hitLayer] && !thisBad)
                  //     keepThis = true;
                  // if((isBad[hitLayer] && thisBad)||(!(isBad[hitLayer] || thisBad)))
                  //   if(thisSize > hitSize[hitLayer])
                  //     keepThis = true;
                  // if(thisBad && !isBad[hitLayer])
                  //     keepThis = false;
                }else
                  keepThis = true;

                if(keepThis)
                  {
                    theHits[hitLayer] = hit;
                    hitSize[hitLayer] = (double)thisSize;
                    isBad[hitLayer] = thisBad;
                    isEdge[hitLayer] = thisEdge;
                    isBig[hitLayer] = thisBad;
                  }
                flagHit[hitLayer] = 1.0;
              }


            }
            // std::cout << "hits" << std::endl;
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
                  x[i] = (double)h->globalState().position.y();
                  y[i] = (double)h->globalState().position.y();
                  z[i] = (double)h->globalState().position.z();
                  phi_hit[i] = (double)h->globalState().phi;
                  r[i] = (double)h->globalState().r;
                  c_x[i] =(double)clust->x();
                  c_y[i] =(double)clust->y();
                  size[i] =(double)clust->size();
                  sizex[i] =(double)clust->sizeX();
                  sizey[i] =(double)clust->sizeY();
                  charge[i] =(double)clust->charge();
                  ovfx[i] =(double)clust->sizeX() > padSize;
                  ovfy[i] =(double)clust->sizeY() > padSize;
                  ratio[i] =(double)(clust->sizeY()) / (double)(clust->sizeX());
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

                  int c = 0;
                  for (int ny = padSize; ny>0; --ny)
                  {
                    for(int nx = 0; nx<padSize; nx++)
                    {

                      int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
                      hitPixels[i][c] = (double)hClust.GetBinContent(n);
                      c++;
                    }
                  }

                }
            }

            // std::cout << "pads" << std::endl;

            theData.push_back(float(tt)); //instead of trackPdg : track number in the collection
            theData.push_back(float(seqNumber_)); //instead of sF: seqNumber in the collection
            theData.push_back(float(trigger));
            theData.push_back(0.0);

            for(int i = 0; i<10;i++)
            {

              theData.push_back((double)x[i]);
              theData.push_back((double)y[i]);
              theData.push_back((double)z[i]);

              theData.push_back((double)phi_hit[i]);
              theData.push_back((double)r[i]);

              theData.push_back((double)c_x[i]);
              theData.push_back((double)c_y[i]);
              theData.push_back((double)size[i]);
              theData.push_back((double)sizex[i]);
              theData.push_back((double)sizey[i]);

              theData.push_back((double)charge[i]);

              theData.push_back((double)ovfx[i]);
              theData.push_back((double)ovfy[i]);

              theData.push_back((double)ratio[i]);

              theData.push_back((double)motherPdgId[i]);
              theData.push_back((double)pdgId[i]);

            }

            for(int i = 0; i<10;i++)
              for(int j =0;j<padSize*padSize;j++)
                theData.push_back((double)(hitPixels[i][j]));

              for (size_t i = 0; i < theData.size(); i++) {
                outPhiFile << theData[i] << "\t";
              }
          }
          outPhiFile << 542.1369 << std::endl;


        } //if vProb > 0.0

           } // loop over second track
         }   // loop on track candidates
}

// ------------ method called once each job just before starting event loop  ------------
void DiTrack::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DiTrack::endJob() {

  std::cout << "=============== Di Track Analyzer ===============" << std::endl;
  std::cout << "n of di tracks: " << nditrak << std::endl;
  std::cout << "=============== ================= ===============" << std::endl;
}

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
