#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/transform.h"

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
#include<type_traits>


#include "TMath.h"
#include <TF1.h>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"

#include <iostream>
#include <string>
#include <fstream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

using namespace std;
using namespace edm;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
namespace {
  bool trackSelected(unsigned char mask, unsigned char qual) {
    return mask & 1<<qual;
  }

}

MultiTrackValidator::MultiTrackValidator(const edm::ParameterSet& pset):
  associators(pset.getUntrackedParameter< std::vector<edm::InputTag> >("associators")),
  label(pset.getParameter< std::vector<edm::InputTag> >("label")),
  parametersDefiner(pset.getParameter<std::string>("parametersDefiner")),
  parametersDefinerIsCosmic_(parametersDefiner == "CosmicParametersDefinerForTP"),
  ignoremissingtkcollection_(pset.getUntrackedParameter<bool>("ignoremissingtrackcollection",false)),
  useAssociators_(pset.getParameter< bool >("UseAssociators")),
  calculateDrSingleCollection_(pset.getUntrackedParameter<bool>("calculateDrSingleCollection")),
  doPlotsOnlyForTruePV_(pset.getUntrackedParameter<bool>("doPlotsOnlyForTruePV")),
  doSummaryPlots_(pset.getUntrackedParameter<bool>("doSummaryPlots")),
  doSimPlots_(pset.getUntrackedParameter<bool>("doSimPlots")),
  doSimTrackPlots_(pset.getUntrackedParameter<bool>("doSimTrackPlots")),
  doRecoTrackPlots_(pset.getUntrackedParameter<bool>("doRecoTrackPlots")),
  dodEdxPlots_(pset.getUntrackedParameter<bool>("dodEdxPlots")),
  doPVAssociationPlots_(pset.getUntrackedParameter<bool>("doPVAssociationPlots")),
  doSeedPlots_(pset.getUntrackedParameter<bool>("doSeedPlots")),
  doMVAPlots_(pset.getUntrackedParameter<bool>("doMVAPlots")),
  simPVMaxZ_(pset.getUntrackedParameter<double>("simPVMaxZ")),
  intHitDoublets_(consumes<IntermediateHitDoublets>(pset.getParameter<edm::InputTag>("doublets"))),
  tpMap_(consumes<ClusterTPAssociation>(pset.getParameter<edm::InputTag>("tpMap")))
{

  padHalfSize = 8;
  padSize = (int)(padHalfSize*2);
  tParams = 22;

  const edm::InputTag& label_tp_effic_tag = pset.getParameter< edm::InputTag >("label_tp_effic");
  const edm::InputTag& label_tp_fake_tag = pset.getParameter< edm::InputTag >("label_tp_fake");

  if(pset.getParameter<bool>("label_tp_effic_refvector")) {
    label_tp_effic_refvector = consumes<TrackingParticleRefVector>(label_tp_effic_tag);
  }
  else {
    label_tp_effic = consumes<TrackingParticleCollection>(label_tp_effic_tag);
  }
  if(pset.getParameter<bool>("label_tp_fake_refvector")) {
    label_tp_fake_refvector = consumes<TrackingParticleRefVector>(label_tp_fake_tag);
  }
  else {
    label_tp_fake = consumes<TrackingParticleCollection>(label_tp_fake_tag);
  }
  label_pileupinfo = consumes<std::vector<PileupSummaryInfo> >(pset.getParameter< edm::InputTag >("label_pileupinfo"));
  for(const auto& tag: pset.getParameter<std::vector<edm::InputTag>>("sim")) {
    simHitTokens_.push_back(consumes<std::vector<PSimHit>>(tag));
  }

  std::vector<edm::InputTag> doResolutionPlotsForLabels = pset.getParameter<std::vector<edm::InputTag> >("doResolutionPlotsForLabels");
  doResolutionPlots_.reserve(label.size());
  for (auto& itag : label) {
    labelToken.push_back(consumes<edm::View<reco::Track> >(itag));
    const bool doResol = doResolutionPlotsForLabels.empty() || (std::find(cbegin(doResolutionPlotsForLabels), cend(doResolutionPlotsForLabels), itag) != cend(doResolutionPlotsForLabels));
    doResolutionPlots_.push_back(doResol);
  }
  { // check for duplicates
    auto labelTmp = edm::vector_transform(label, [&](const edm::InputTag& tag) { return tag.label(); });
    std::sort(begin(labelTmp), end(labelTmp));
    std::string empty;
    const std::string* prev = &empty;
    for(const std::string& l: labelTmp) {
      if(l == *prev) {
        throw cms::Exception("Configuration") << "Duplicate InputTag in labels: " << l;
      }
      prev = &l;
    }
  }

  edm::InputTag beamSpotTag = pset.getParameter<edm::InputTag>("beamSpot");
  bsSrc = consumes<reco::BeamSpot>(beamSpotTag);

  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  histoProducerAlgo_ = std::make_unique<MTVHistoProducerAlgoForTracker>(psetForHistoProducerAlgo, beamSpotTag, doSeedPlots_, consumesCollector());

  dirName_ = pset.getParameter<std::string>("dirName");

  tpNLayersToken_ = consumes<edm::ValueMap<unsigned int> >(pset.getParameter<edm::InputTag>("label_tp_nlayers"));
  tpNPixelLayersToken_ = consumes<edm::ValueMap<unsigned int> >(pset.getParameter<edm::InputTag>("label_tp_npixellayers"));
  tpNStripStereoLayersToken_ = consumes<edm::ValueMap<unsigned int> >(pset.getParameter<edm::InputTag>("label_tp_nstripstereolayers"));

  if(dodEdxPlots_) {
    m_dEdx1Tag = consumes<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx1Tag"));
    m_dEdx2Tag = consumes<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx2Tag"));
  }

  label_tv = consumes<TrackingVertexCollection>(pset.getParameter< edm::InputTag >("label_tv"));
  if(doPlotsOnlyForTruePV_ || doPVAssociationPlots_) {
    recoVertexToken_ = consumes<edm::View<reco::Vertex> >(pset.getUntrackedParameter<edm::InputTag>("label_vertex"));
    vertexAssociatorToken_ = consumes<reco::VertexToTrackingVertexAssociator>(pset.getUntrackedParameter<edm::InputTag>("vertexAssociator"));
  }

  if(doMVAPlots_) {
    mvaQualityCollectionTokens_.resize(labelToken.size());
    auto mvaPSet = pset.getUntrackedParameter<edm::ParameterSet>("mvaLabels");
    for(size_t iIter=0; iIter<labelToken.size(); ++iIter) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(labelToken[iIter], labels);
      if(mvaPSet.exists(labels.module)) {
        mvaQualityCollectionTokens_[iIter] = edm::vector_transform(mvaPSet.getUntrackedParameter<std::vector<std::string> >(labels.module),
                                                                   [&](const std::string& tag) {
                                                                     return std::make_tuple(consumes<MVACollection>(edm::InputTag(tag, "MVAValues")),
                                                                                            consumes<QualityMaskCollection>(edm::InputTag(tag, "QualityMasks")));
                                                                   });
      }
    }
  }

  tpSelector = TrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
                                        pset.getParameter<double>("ptMaxTP"),
					pset.getParameter<double>("minRapidityTP"),
					pset.getParameter<double>("maxRapidityTP"),
					pset.getParameter<double>("tipTP"),
					pset.getParameter<double>("lipTP"),
					pset.getParameter<int>("minHitTP"),
					pset.getParameter<bool>("signalOnlyTP"),
					pset.getParameter<bool>("intimeOnlyTP"),
					pset.getParameter<bool>("chargedOnlyTP"),
					pset.getParameter<bool>("stableOnlyTP"),
					pset.getParameter<std::vector<int> >("pdgIdTP"));

  cosmictpSelector = CosmicTrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
						    pset.getParameter<double>("minRapidityTP"),
						    pset.getParameter<double>("maxRapidityTP"),
						    pset.getParameter<double>("tipTP"),
						    pset.getParameter<double>("lipTP"),
						    pset.getParameter<int>("minHitTP"),
						    pset.getParameter<bool>("chargedOnlyTP"),
						    pset.getParameter<std::vector<int> >("pdgIdTP"));


  ParameterSet psetVsPhi = psetForHistoProducerAlgo.getParameter<ParameterSet>("TpSelectorForEfficiencyVsPhi");
  dRtpSelector = TrackingParticleSelector(psetVsPhi.getParameter<double>("ptMin"),
                                          psetVsPhi.getParameter<double>("ptMax"),
					  psetVsPhi.getParameter<double>("minRapidity"),
					  psetVsPhi.getParameter<double>("maxRapidity"),
					  psetVsPhi.getParameter<double>("tip"),
					  psetVsPhi.getParameter<double>("lip"),
					  psetVsPhi.getParameter<int>("minHit"),
					  psetVsPhi.getParameter<bool>("signalOnly"),
					  psetVsPhi.getParameter<bool>("intimeOnly"),
					  psetVsPhi.getParameter<bool>("chargedOnly"),
					  psetVsPhi.getParameter<bool>("stableOnly"),
					  psetVsPhi.getParameter<std::vector<int> >("pdgId"));

  dRTrackSelector = MTVHistoProducerAlgoForTracker::makeRecoTrackSelectorFromTPSelectorParameters(psetVsPhi, beamSpotTag, consumesCollector());

  useGsf = pset.getParameter<bool>("useGsf");

  _simHitTpMapTag = mayConsume<SimHitTPAssociationProducer::SimHitTPAssociationList>(pset.getParameter<edm::InputTag>("simHitTpMapTag"));

  if(calculateDrSingleCollection_) {
    labelTokenForDrCalculation = consumes<edm::View<reco::Track> >(pset.getParameter<edm::InputTag>("trackCollectionForDrCalculation"));
  }

  if(useAssociators_) {
    for (auto const& src: associators) {
      associatorTokens.push_back(consumes<reco::TrackToTrackingParticleAssociator>(src));
    }
  } else {
    for (auto const& src: associators) {
      associatormapStRs.push_back(consumes<reco::SimToRecoCollection>(src));
      associatormapRtSs.push_back(consumes<reco::RecoToSimCollection>(src));
    }
  }
}


MultiTrackValidator::~MultiTrackValidator() {}


void MultiTrackValidator::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const& setup) {

  const auto minColl = -0.5;
  const auto maxColl = label.size()-0.5;
  const auto nintColl = label.size();

  auto binLabels = [&](MonitorElement *me) {
    TH1 *h = me->getTH1();
    for(size_t i=0; i<label.size(); ++i) {
      h->GetXaxis()->SetBinLabel(i+1, label[i].label().c_str());
    }
    return me;
  };

  //Booking histograms concerning with simulated tracks
  if(doSimPlots_) {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + "simulation");

    histoProducerAlgo_->bookSimHistos(ibook);

    ibook.cd();
    ibook.setCurrentFolder(dirName_);
  }

  for (unsigned int ww=0;ww<associators.size();ww++){
    ibook.cd();
    // FIXME: these need to be moved to a subdirectory whose name depends on the associator
    ibook.setCurrentFolder(dirName_);

    if(doSummaryPlots_) {
      if(doSimTrackPlots_) {
        h_assoc_coll.push_back(binLabels( ibook.book1D("num_assoc(simToReco)_coll", "N of associated (simToReco) tracks vs track collection", nintColl, minColl, maxColl) ));
        h_simul_coll.push_back(binLabels( ibook.book1D("num_simul_coll", "N of simulated tracks vs track collection", nintColl, minColl, maxColl) ));
      }
      if(doRecoTrackPlots_) {
        h_reco_coll.push_back(binLabels( ibook.book1D("num_reco_coll", "N of reco track vs track collection", nintColl, minColl, maxColl) ));
        h_assoc2_coll.push_back(binLabels( ibook.book1D("num_assoc(recoToSim)_coll", "N of associated (recoToSim) tracks vs track collection", nintColl, minColl, maxColl) ));
        h_looper_coll.push_back(binLabels( ibook.book1D("num_duplicate_coll", "N of associated (recoToSim) looper tracks vs track collection", nintColl, minColl, maxColl) ));
        h_pileup_coll.push_back(binLabels( ibook.book1D("num_pileup_coll", "N of associated (recoToSim) pileup tracks vs track collection", nintColl, minColl, maxColl) ));
      }
    }

    for (unsigned int www=0;www<label.size();www++){
      ibook.cd();
      InputTag algo = label[www];
      string dirName=dirName_;
      if (algo.process()!="")
        dirName+=algo.process()+"_";
      if(algo.label()!="")
        dirName+=algo.label()+"_";
      if(algo.instance()!="")
        dirName+=algo.instance()+"_";
      if (dirName.find("Tracks")<dirName.length()){
        dirName.replace(dirName.find("Tracks"),6,"");
      }
      string assoc= associators[ww].label();
      if (assoc.find("Track")<assoc.length()){
        assoc.replace(assoc.find("Track"),5,"");
      }
      dirName+=assoc;
      std::replace(dirName.begin(), dirName.end(), ':', '_');

      ibook.setCurrentFolder(dirName);

      const bool doResolutionPlots = doResolutionPlots_[www];

      if(doSimTrackPlots_) {
        histoProducerAlgo_->bookSimTrackHistos(ibook, doResolutionPlots);
        if(doPVAssociationPlots_) histoProducerAlgo_->bookSimTrackPVAssociationHistos(ibook);
      }

      //Booking histograms concerning with reconstructed tracks
      if(doRecoTrackPlots_) {
        histoProducerAlgo_->bookRecoHistos(ibook, doResolutionPlots);
        if (dodEdxPlots_) histoProducerAlgo_->bookRecodEdxHistos(ibook);
        if (doPVAssociationPlots_) histoProducerAlgo_->bookRecoPVAssociationHistos(ibook);
        if (doMVAPlots_) histoProducerAlgo_->bookMVAHistos(ibook, mvaQualityCollectionTokens_[www].size());
      }

      if(doSeedPlots_) {
        histoProducerAlgo_->bookSeedHistos(ibook);
      }
    }//end loop www
  }// end loop ww
}

namespace {
  void ensureEffIsSubsetOfFake(const TrackingParticleRefVector& eff, const TrackingParticleRefVector& fake) {
    // If efficiency RefVector is empty, don't check the product ids
    // as it will be 0:0 for empty. This covers also the case where
    // both are empty. The case of fake being empty and eff not is an
    // error.
    if(eff.empty())
      return;

    // First ensure product ids
    if(eff.id() != fake.id()) {
      throw cms::Exception("Configuration") << "Efficiency and fake TrackingParticle (refs) point to different collections (eff " << eff.id() << " fake " << fake.id() << "). This is not supported. Efficiency TP set must be the same or a subset of the fake TP set.";
    }

    // Same technique as in associationMapFilterValues
    edm::IndexSet fakeKeys;
    fakeKeys.reserve(fake.size());
    for(const auto& ref: fake) {
      fakeKeys.insert(ref.key());
    }

    for(const auto& ref: eff) {
      if(!fakeKeys.has(ref.key())) {
        throw cms::Exception("Configuration") << "Efficiency TrackingParticle " << ref.key() << " is not found from the set of fake TPs. This is not supported. The efficiency TP set must be the same or a subset of the fake TP set.";
      }
    }
  }
}

const TrackingVertex::LorentzVector *MultiTrackValidator::getSimPVPosition(const edm::Handle<TrackingVertexCollection>& htv) const {
  for(const auto& simV: *htv) {
    if(simV.eventId().bunchCrossing() != 0) continue; // remove OOTPU
    if(simV.eventId().event() != 0) continue; // pick the PV of hard scatter
    return &(simV.position());
  }
  return nullptr;
}

const reco::Vertex::Point *MultiTrackValidator::getRecoPVPosition(const edm::Event& event, const edm::Handle<TrackingVertexCollection>& htv) const {
  edm::Handle<edm::View<reco::Vertex> > hvertex;
  event.getByToken(recoVertexToken_, hvertex);

  edm::Handle<reco::VertexToTrackingVertexAssociator> hvassociator;
  event.getByToken(vertexAssociatorToken_, hvassociator);

  auto v_r2s = hvassociator->associateRecoToSim(hvertex, htv);
  auto pvPtr = hvertex->refAt(0);
  if(pvPtr->isFake() || pvPtr->ndof() < 0) // skip junk vertices
    return nullptr;

  auto pvFound = v_r2s.find(pvPtr);
  if(pvFound == v_r2s.end())
    return nullptr;

  for(const auto& vertexRefQuality: pvFound->val) {
    const TrackingVertex& tv = *(vertexRefQuality.first);
    if(tv.eventId().event() == 0 && tv.eventId().bunchCrossing() == 0) {
      return &(pvPtr->position());
    }
  }

  return nullptr;
}

void MultiTrackValidator::tpParametersAndSelection(const TrackingParticleRefVector& tPCeff,
                                                   const ParametersDefinerForTP& parametersDefinerTP,
                                                   const edm::Event& event, const edm::EventSetup& setup,
                                                   const reco::BeamSpot& bs,
                                                   std::vector<std::tuple<TrackingParticle::Vector, TrackingParticle::Point> >& momVert_tPCeff,
                                                   std::vector<size_t>& selected_tPCeff) const {
  selected_tPCeff.reserve(tPCeff.size());
  momVert_tPCeff.reserve(tPCeff.size());
  int nIntimeTPs = 0;
  if(parametersDefinerIsCosmic_) {
    for(size_t j=0; j<tPCeff.size(); ++j) {
      const TrackingParticleRef& tpr = tPCeff[j];

      TrackingParticle::Vector momentum = parametersDefinerTP.momentum(event,setup,tpr);
      TrackingParticle::Point vertex = parametersDefinerTP.vertex(event,setup,tpr);
      if(doSimPlots_) {
        histoProducerAlgo_->fill_generic_simTrack_histos(momentum, vertex, tpr->eventId().bunchCrossing());
      }
      if(tpr->eventId().bunchCrossing() == 0)
        ++nIntimeTPs;

      if(cosmictpSelector(tpr,&bs,event,setup)) {
        selected_tPCeff.push_back(j);
        momVert_tPCeff.emplace_back(momentum, vertex);
      }
    }
  }
  else {
    size_t j=0;
    for(auto const& tpr: tPCeff) {
      const TrackingParticle& tp = *tpr;

      // TODO: do we want to fill these from all TPs that include IT
      // and OOT (as below), or limit to IT+OOT TPs passing tpSelector
      // (as it was before)? The latter would require another instance
      // of tpSelector with intimeOnly=False.
      if(doSimPlots_) {
        histoProducerAlgo_->fill_generic_simTrack_histos(tp.momentum(), tp.vertex(), tp.eventId().bunchCrossing());
      }
      if(tp.eventId().bunchCrossing() == 0)
        ++nIntimeTPs;

      if(tpSelector(tp)) {
        selected_tPCeff.push_back(j);
        TrackingParticle::Vector momentum = parametersDefinerTP.momentum(event,setup,tpr);
        TrackingParticle::Point vertex = parametersDefinerTP.vertex(event,setup,tpr);
        momVert_tPCeff.emplace_back(momentum, vertex);
      }
      ++j;
    }
  }
  if(doSimPlots_) {
    histoProducerAlgo_->fill_simTrackBased_histos(nIntimeTPs);
  }
}


size_t MultiTrackValidator::tpDR(const TrackingParticleRefVector& tPCeff,
                                 const std::vector<size_t>& selected_tPCeff,
                                 DynArray<float>& dR_tPCeff) const {
  float etaL[tPCeff.size()], phiL[tPCeff.size()];
  size_t n_selTP_dr = 0;
  for(size_t iTP: selected_tPCeff) {
    //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
    auto const& tp2 = *(tPCeff[iTP]);
    auto  && p = tp2.momentum();
    etaL[iTP] = etaFromXYZ(p.x(),p.y(),p.z());
    phiL[iTP] = atan2f(p.y(),p.x());
  }
  for(size_t iTP1: selected_tPCeff) {
    auto const& tp = *(tPCeff[iTP1]);
    double dR = std::numeric_limits<double>::max();
    if(dRtpSelector(tp)) {//only for those needed for efficiency!
      ++n_selTP_dr;
      float eta = etaL[iTP1];
      float phi = phiL[iTP1];
      for(size_t iTP2: selected_tPCeff) {
        //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
        if (iTP1==iTP2) {continue;}
        auto dR_tmp = reco::deltaR2(eta, phi, etaL[iTP2], phiL[iTP2]);
        if (dR_tmp<dR) dR=dR_tmp;
      }  // ttp2 (iTP)
    }
    dR_tPCeff[iTP1] = std::sqrt(dR);
  }  // tp
  return n_selTP_dr;
}

void MultiTrackValidator::trackDR(const edm::View<reco::Track>& trackCollection, const edm::View<reco::Track>& trackCollectionDr, DynArray<float>& dR_trk) const {
  int i=0;
  float etaL[trackCollectionDr.size()];
  float phiL[trackCollectionDr.size()];
  bool validL[trackCollectionDr.size()];
  for (auto const & track2 : trackCollectionDr) {
    auto  && p = track2.momentum();
    etaL[i] = etaFromXYZ(p.x(),p.y(),p.z());
    phiL[i] = atan2f(p.y(),p.x());
    validL[i] = !trackFromSeedFitFailed(track2);
    ++i;
  }
  for(View<reco::Track>::size_type i=0; i<trackCollection.size(); ++i){
    auto const &  track = trackCollection[i];
    auto dR = std::numeric_limits<float>::max();
    if(!trackFromSeedFitFailed(track)) {
      auto  && p = track.momentum();
      float eta = etaFromXYZ(p.x(),p.y(),p.z());
      float phi = atan2f(p.y(),p.x());
      for(View<reco::Track>::size_type j=0; j<trackCollectionDr.size(); ++j){
        if(!validL[j]) continue;
        auto dR_tmp = reco::deltaR2(eta, phi, etaL[j], phiL[j]);
        if ( (dR_tmp<dR) & (dR_tmp>std::numeric_limits<float>::min())) dR=dR_tmp;
      }
    }
    dR_trk[i] = std::sqrt(dR);
  }
}


void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){
  using namespace reco;

  LogDebug("TrackValidator") << "\n====================================================" << "\n"
                             << "Analyzing new event" << "\n"
                             << "====================================================\n" << "\n";


  edm::ESHandle<ParametersDefinerForTP> parametersDefinerTPHandle;
  setup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTPHandle);
  //Since we modify the object, we must clone it
  auto parametersDefinerTP = parametersDefinerTPHandle->clone();

  std::vector < IntermediateHitDoublets> allDoublets;
  std::vector < std::string > allDoubletsNames;
  std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31};

  edm::Handle<IntermediateHitDoublets> detachedQuadStepHitDoublets;
  event.getByToken(detachedQuadStepHitDoublets_,detachedQuadStepHitDoublets);
  allDoublets.push_back(detachedQuadStepHitDoublets);
  allDoubletsNames.push_back("detachedQuadStepHitDoublets");

  edm::Handle<IntermediateHitDoublets> detachedTripletStepHitDoublets;
  event.getByToken(detachedTripletStepHitDoublets_,detachedTripletStepHitDoublets);
  allDoublets.push_back(detachedTripletStepHitDoublets);
  allDoubletsNames.push_back("detachedTripletStepHitDoublets");

  edm::Handle<IntermediateHitDoublets> initialStepHitDoublets;
  event.getByToken(initialStepHitDoublets_,initialStepHitDoublets);
  allDoublets.push_back(initialStepHitDoublets);
  allDoubletsNames.push_back("initialStepHitDoublets");

  edm::Handle<IntermediateHitDoublets> lowPtQuadStepHitDoublets;
  event.getByToken(lowPtQuadStepHitDoublets_,lowPtQuadStepHitDoublets);
  allDoublets.push_back(lowPtQuadStepHitDoublets);
  allDoubletsNames.push_back("lowPtQuadStepHitDoublets");

  edm::Handle<IntermediateHitDoublets> mixedTripletStepHitDoubletsA;
  event.getByToken(mixedTripletStepHitDoubletsA_,mixedTripletStepHitDoubletsA);
  allDoublets.push_back(mixedTripletStepHitDoubletsA);
  allDoubletsNames.push_back("mixedTripletStepHitDoubletsA");

  edm::Handle<IntermediateHitDoublets> mixedTripletStepHitDoubletsB;
  event.getByToken(mixedTripletStepHitDoubletsB_,mixedTripletStepHitDoubletsB);
  allDoublets.push_back(mixedTripletStepHitDoubletsB);
  allDoubletsNames.push_back("mixedTripletStepHitDoubletsB");

  edm::Handle<IntermediateHitDoublets> pixelLessStepHitDoublets;
  event.getByToken(pixelLessStepHitDoublets_,pixelLessStepHitDoublets);
  allDoublets.push_back(pixelLessStepHitDoublets);
  allDoubletsNames.push_back("pixelLessStepHitDoublets");

  edm::Handle<IntermediateHitDoublets> tripletElectronHitDoublets;
  event.getByToken(tripletElectronHitDoublets_,tripletElectronHitDoublets);
  allDoublets.push_back(tripletElectronHitDoublets);
  allDoubletsNames.push_back("tripletElectronHitDoublets");


  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;

  edm::Handle<ClusterTPAssociation> tpClust;
  event.getByToken(tpMap_,tpClust);

  // FIXME: we really need to move to edm::View for reading the
  // TrackingParticles... Unfortunately it has non-trivial
  // consequences on the associator/association interfaces etc.
  TrackingParticleRefVector tmpTPeff;
  TrackingParticleRefVector tmpTPfake;
  const TrackingParticleRefVector *tmpTPeffPtr = nullptr;
  const TrackingParticleRefVector *tmpTPfakePtr = nullptr;

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff;
  edm::Handle<TrackingParticleRefVector>  TPCollectionHeffRefVector;

  const bool tp_effic_refvector = label_tp_effic.isUninitialized();
  if(!tp_effic_refvector) {
    event.getByToken(label_tp_effic, TPCollectionHeff);
    for(size_t i=0, size=TPCollectionHeff->size(); i<size; ++i) {
      tmpTPeff.push_back(TrackingParticleRef(TPCollectionHeff, i));
    }
    tmpTPeffPtr = &tmpTPeff;
  }
  else {
    event.getByToken(label_tp_effic_refvector, TPCollectionHeffRefVector);
    tmpTPeffPtr = TPCollectionHeffRefVector.product();
  }
  if(!label_tp_fake.isUninitialized()) {
    edm::Handle<TrackingParticleCollection> TPCollectionHfake ;
    event.getByToken(label_tp_fake,TPCollectionHfake);
    for(size_t i=0, size=TPCollectionHfake->size(); i<size; ++i) {
      tmpTPfake.push_back(TrackingParticleRef(TPCollectionHfake, i));
    }
    tmpTPfakePtr = &tmpTPfake;
  }
  else {
    edm::Handle<TrackingParticleRefVector> TPCollectionHfakeRefVector;
    event.getByToken(label_tp_fake_refvector, TPCollectionHfakeRefVector);
    tmpTPfakePtr = TPCollectionHfakeRefVector.product();
  }

  TrackingParticleRefVector const & tPCeff = *tmpTPeffPtr;
  TrackingParticleRefVector const & tPCfake = *tmpTPfakePtr;

  ensureEffIsSubsetOfFake(tPCeff, tPCfake);

  if(parametersDefinerIsCosmic_) {
    edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
    //warning: make sure the TP collection used in the map is the same used in the MTV!
    event.getByToken(_simHitTpMapTag,simHitsTPAssoc);
    parametersDefinerTP->initEvent(simHitsTPAssoc);
    cosmictpSelector.initEvent(simHitsTPAssoc);
  }
  dRTrackSelector->init(event, setup);
  histoProducerAlgo_->init(event, setup);

  // Find the sim PV and tak its position
  edm::Handle<TrackingVertexCollection> htv;
  event.getByToken(label_tv, htv);
  const TrackingVertex::LorentzVector *theSimPVPosition = getSimPVPosition(htv);
  if(simPVMaxZ_ >= 0) {
    if(!theSimPVPosition) return;
    if(std::abs(theSimPVPosition->z()) > simPVMaxZ_) return;
  }

  // Check, when necessary, if reco PV matches to sim PV
  const reco::Vertex::Point *thePVposition = nullptr;
  if(doPlotsOnlyForTruePV_ || doPVAssociationPlots_) {
    thePVposition = getRecoPVPosition(event, htv);
    if(doPlotsOnlyForTruePV_ && !thePVposition)
      return;

    // Rest of the code assumes that if thePVposition is non-null, the
    // PV-association histograms get filled. In above, the "nullness"
    // is used to deliver the information if the reco PV is matched to
    // the sim PV.
    if(!doPVAssociationPlots_)
      thePVposition = nullptr;
  }

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByToken(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot const & bs = *recoBeamSpotHandle;

  edm::Handle< std::vector<PileupSummaryInfo> > puinfoH;
  event.getByToken(label_pileupinfo,puinfoH);
  PileupSummaryInfo puinfo;

  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }

  // Number of 3D layers for TPs
  edm::Handle<edm::ValueMap<unsigned int>> tpNLayersH;
  event.getByToken(tpNLayersToken_, tpNLayersH);
  const auto& nLayers_tPCeff = *tpNLayersH;

  event.getByToken(tpNPixelLayersToken_, tpNLayersH);
  const auto& nPixelLayers_tPCeff = *tpNLayersH;

  event.getByToken(tpNStripStereoLayersToken_, tpNLayersH);
  const auto& nStripMonoAndStereoLayers_tPCeff = *tpNLayersH;

  // Precalculate TP selection (for efficiency), and momentum and vertex wrt PCA
  //
  // TODO: ParametersDefinerForTP ESProduct needs to be changed to
  // EDProduct because of consumes.
  //
  // In principle, we could just precalculate the momentum and vertex
  // wrt PCA for all TPs for once and put that to the event. To avoid
  // repetitive calculations those should be calculated only once for
  // each TP. That would imply that we should access TPs via Refs
  // (i.e. View) in here, since, in general, the eff and fake TP
  // collections can be different (and at least HI seems to use that
  // feature). This would further imply that the
  // RecoToSimCollection/SimToRecoCollection should be changed to use
  // View<TP> instead of vector<TP>, and migrate everything.
  //
  // Or we could take only one input TP collection, and do another
  // TP-selection to obtain the "fake" collection like we already do
  // for "efficiency" TPs.
  std::vector<size_t> selected_tPCeff;
  std::vector<std::tuple<TrackingParticle::Vector, TrackingParticle::Point>> momVert_tPCeff;
  tpParametersAndSelection(tPCeff, *parametersDefinerTP, event, setup, bs, momVert_tPCeff, selected_tPCeff);

  //calculate dR for TPs
  declareDynArray(float, tPCeff.size(), dR_tPCeff);
  size_t n_selTP_dr = tpDR(tPCeff, selected_tPCeff, dR_tPCeff);

  edm::Handle<View<Track> >  trackCollectionForDrCalculation;
  if(calculateDrSingleCollection_) {
    event.getByToken(labelTokenForDrCalculation, trackCollectionForDrCalculation);
  }

  // dE/dx
  // at some point this could be generalized, with a vector of tags and a corresponding vector of Handles
  // I'm writing the interface such to take vectors of ValueMaps
  std::vector<const edm::ValueMap<reco::DeDxData> *> v_dEdx;
  if(dodEdxPlots_) {
    edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx1Handle;
    edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx2Handle;
    event.getByToken(m_dEdx1Tag, dEdx1Handle);
    event.getByToken(m_dEdx2Tag, dEdx2Handle);
    v_dEdx.push_back(dEdx1Handle.product());
    v_dEdx.push_back(dEdx2Handle.product());
  }

  std::vector<const MVACollection *> mvaCollections;
  std::vector<const QualityMaskCollection *> qualityMaskCollections;
  std::vector<float> mvaValues;

  int w=0; //counter counting the number of sets of histograms
  for (unsigned int ww=0;ww<associators.size();ww++){
    // run value filtering of recoToSim map already here as it depends only on the association, not track collection
    reco::SimToRecoCollection const * simRecCollPFull=nullptr;
    reco::RecoToSimCollection const * recSimCollP=nullptr;
    reco::RecoToSimCollection recSimCollL;
    if(!useAssociators_) {
      Handle<reco::SimToRecoCollection > simtorecoCollectionH;
      event.getByToken(associatormapStRs[ww], simtorecoCollectionH);
      simRecCollPFull = simtorecoCollectionH.product();

      Handle<reco::RecoToSimCollection > recotosimCollectionH;
      event.getByToken(associatormapRtSs[ww],recotosimCollectionH);
      recSimCollP = recotosimCollectionH.product();

      // We need to filter the associations of the fake-TrackingParticle
      // collection only from RecoToSim collection, otherwise the
      // RecoToSim histograms get false entries
      recSimCollL = associationMapFilterValues(*recSimCollP, tPCfake);
      recSimCollP = &recSimCollL;
    }

    for (unsigned int www=0;www<label.size();www++, w++){ // need to increment w here, since there are many continues in the loop body
      //
      //get collections from the event
      //
      edm::Handle<View<Track> >  trackCollectionHandle;
      if(!event.getByToken(labelToken[www], trackCollectionHandle)&&ignoremissingtkcollection_)continue;
      const edm::View<Track>& trackCollection = *trackCollectionHandle;

      reco::SimToRecoCollection const * simRecCollP=nullptr;
      reco::SimToRecoCollection simRecCollL;

      //associate tracks
      LogTrace("TrackValidator") << "Analyzing "
                                 << label[www] << " with "
                                 << associators[ww] <<"\n";
      if(useAssociators_){
        edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
        event.getByToken(associatorTokens[ww], theAssociator);

        // The associator interfaces really need to be fixed...
        edm::RefToBaseVector<reco::Track> trackRefs;
        for(edm::View<Track>::size_type i=0; i<trackCollection.size(); ++i) {
          trackRefs.push_back(trackCollection.refAt(i));
        }


	LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
        recSimCollL = std::move(theAssociator->associateRecoToSim(trackRefs, tPCfake));
        recSimCollP = &recSimCollL;
	LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
        // It is necessary to do the association wrt. fake TPs,
        // because this SimToReco association is used also for
        // duplicates. Since the set of efficiency TPs are required to
        // be a subset of the set of fake TPs, for efficiency
        // histograms it doesn't matter if the association contains
        // associations of TPs not in the set of efficiency TPs.
        simRecCollL = std::move(theAssociator->associateSimToReco(trackRefs, tPCfake));
        simRecCollP = &simRecCollL;
      }
      else{
        // We need to filter the associations of the current track
        // collection only from SimToReco collection, otherwise the
        // SimToReco histograms get false entries. The filtering must
        // be done separately for each track collection.
        simRecCollL = associationMapFilterValues(*simRecCollPFull, trackCollection);
        simRecCollP = &simRecCollL;
      }

      reco::RecoToSimCollection const & recSimColl = *recSimCollP;
      reco::SimToRecoCollection const & simRecColl = *simRecCollP;

      // read MVA collections
      if(doMVAPlots_ && !mvaQualityCollectionTokens_[www].empty()) {
        edm::Handle<MVACollection> hmva;
        edm::Handle<QualityMaskCollection> hqual;
        for(const auto& tokenTpl: mvaQualityCollectionTokens_[www]) {
          event.getByToken(std::get<0>(tokenTpl), hmva);
          event.getByToken(std::get<1>(tokenTpl), hqual);

          mvaCollections.push_back(hmva.product());
          qualityMaskCollections.push_back(hqual.product());
          if(mvaCollections.back()->size() != trackCollection.size()) {
            throw cms::Exception("Configuration") << "Inconsistency in track collection and MVA sizes. Track collection " << www << " has " << trackCollection.size() << " tracks, whereas the MVA " << (mvaCollections.size()-1) << " for it has " << mvaCollections.back()->size() << " entries. Double-check your configuration.";
          }
          if(qualityMaskCollections.back()->size() != trackCollection.size()) {
            throw cms::Exception("Configuration") << "Inconsistency in track collection and quality mask sizes. Track collection " << www << " has " << trackCollection.size() << " tracks, whereas the quality mask " << (qualityMaskCollections.size()-1) << " for it has " << qualityMaskCollections.back()->size() << " entries. Double-check your configuration.";
          }
        }
      }

      // ########################################################
      // fill simulation histograms (LOOP OVER TRACKINGPARTICLES)
      // ########################################################

      //compute number of tracks per eta interval
      //
      LogTrace("TrackValidator") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      int ats(0);  	  //This counter counts the number of simTracks that are "associated" to recoTracks
      int st(0);    	  //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )

      //loop over already-selected TPs for tracking efficiency
      for(size_t i=0; i<selected_tPCeff.size(); ++i) {
        size_t iTP = selected_tPCeff[i];
        const TrackingParticleRef& tpr = tPCeff[iTP];
        const TrackingParticle& tp = *tpr;

        auto const& momVert = momVert_tPCeff[i];
	TrackingParticle::Vector momentumTP;
	TrackingParticle::Point vertexTP;

	double dxySim(0);
	double dzSim(0);
        double dxyPVSim = 0;
        double dzPVSim = 0;
	double dR=dR_tPCeff[iTP];

	//---------- THIS PART HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------
	//If the TrackingParticle is collison like, get the momentum and vertex at production state
	if(!parametersDefinerIsCosmic_)
	  {
	    momentumTP = tp.momentum();
	    vertexTP = tp.vertex();
	    //Calcualte the impact parameters w.r.t. PCA
	    const TrackingParticle::Vector& momentum = std::get<TrackingParticle::Vector>(momVert);
	    const TrackingParticle::Point& vertex = std::get<TrackingParticle::Point>(momVert);
	    dxySim = TrackingParticleIP::dxy(vertex, momentum, bs.position());
	    dzSim = TrackingParticleIP::dz(vertex, momentum, bs.position());

            if(theSimPVPosition) {
              dxyPVSim = TrackingParticleIP::dxy(vertex, momentum, *theSimPVPosition);
              dzPVSim = TrackingParticleIP::dz(vertex, momentum, *theSimPVPosition);
            }
	  }
	//If the TrackingParticle is comics, get the momentum and vertex at PCA
	else
	  {
	    momentumTP = std::get<TrackingParticle::Vector>(momVert);
	    vertexTP = std::get<TrackingParticle::Point>(momVert);
	    dxySim = TrackingParticleIP::dxy(vertexTP, momentumTP, bs.position());
	    dzSim = TrackingParticleIP::dz(vertexTP, momentumTP, bs.position());

            // Do dxy and dz vs. PV make any sense for cosmics? I guess not
	  }
	//---------- THE PART ABOVE HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------

	// in the coming lines, histos are filled using as input
	// - momentumTP
	// - vertexTP
	// - dxySim
	// - dzSim
        if(!doSimTrackPlots_)
          continue;

	// ##############################################
	// fill RecoAssociated SimTracks' histograms
	// ##############################################
	const reco::Track *matchedTrackPointer = nullptr;
	const reco::Track *matchedSecondTrackPointer = nullptr;
        unsigned int selectsLoose = mvaCollections.size();
        unsigned int selectsHP = mvaCollections.size();
	if(simRecColl.find(tpr) != simRecColl.end()){
	  auto const & rt = simRecColl[tpr];
	  if (!rt.empty()) {
	    ats++; //This counter counts the number of simTracks that have a recoTrack associated
	    // isRecoMatched = true; // UNUSED
	    matchedTrackPointer = rt.begin()->first.get();
	    if(rt.size() >= 2) {
	      matchedSecondTrackPointer = (rt.begin()+1)->first.get();
	    }
	    LogTrace("TrackValidator") << "TrackingParticle #" << st
                                       << " with pt=" << sqrt(momentumTP.perp2())
                                       << " associated with quality:" << rt.begin()->second <<"\n";

            if(doMVAPlots_) {
              // for each MVA we need to take the value of the track
              // with largest MVA value (for the cumulative histograms)
              //
              // also identify the first MVA that possibly selects any
              // track matched to this TrackingParticle, separately
              // for loose and highPurity qualities
              for(size_t imva=0; imva<mvaCollections.size(); ++imva) {
                const auto& mva = *(mvaCollections[imva]);
                const auto& qual = *(qualityMaskCollections[imva]);

                auto iMatch = rt.begin();
                float maxMva = mva[iMatch->first.key()];
                for(; iMatch!=rt.end(); ++iMatch) {
                  auto itrk = iMatch->first.key();
                  maxMva = std::max(maxMva, mva[itrk]);

                  if(selectsLoose >= imva && trackSelected(qual[itrk], reco::TrackBase::loose))
                    selectsLoose = imva;
                  if(selectsHP >= imva && trackSelected(qual[itrk], reco::TrackBase::highPurity))
                    selectsHP = imva;
                }
                mvaValues.push_back(maxMva);
              }
            }
	  }
	}else{
	  LogTrace("TrackValidator")
	    << "TrackingParticle #" << st
	    << " with pt,eta,phi: "
	    << sqrt(momentumTP.perp2()) << " , "
	    << momentumTP.eta() << " , "
	    << momentumTP.phi() << " , "
	    << " NOT associated to any reco::Track" << "\n";
	}




        int nSimHits = tp.numberOfTrackerHits();
        int nSimLayers = nLayers_tPCeff[tpr];
        int nSimPixelLayers = nPixelLayers_tPCeff[tpr];
        int nSimStripMonoAndStereoLayers = nStripMonoAndStereoLayers_tPCeff[tpr];
        histoProducerAlgo_->fill_recoAssociated_simTrack_histos(w,tp,momentumTP,vertexTP,dxySim,dzSim,dxyPVSim,dzPVSim,nSimHits,nSimLayers,nSimPixelLayers,nSimStripMonoAndStereoLayers,matchedTrackPointer,puinfo.getPU_NumInteractions(), dR, thePVposition, theSimPVPosition, bs.position(), mvaValues, selectsLoose, selectsHP);
        mvaValues.clear();

        if(matchedTrackPointer && matchedSecondTrackPointer) {
          histoProducerAlgo_->fill_duplicate_histos(w, *matchedTrackPointer, *matchedSecondTrackPointer);
        }

          if(doSummaryPlots_) {
            if(dRtpSelector(tp)) {
              h_simul_coll[ww]->Fill(www);
              if (matchedTrackPointer) {
                h_assoc_coll[ww]->Fill(www);
              }
            }
          }




      } // End  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){

      // ##############################################
      // fill recoTracks histograms (LOOP OVER TRACKS)
      // ##############################################
      if(!doRecoTrackPlots_)
        continue;
      LogTrace("TrackValidator") << "\n# of reco::Tracks with "
                                 << label[www].process()<<":"
                                 << label[www].label()<<":"
                                 << label[www].instance()
                                 << ": " << trackCollection.size() << "\n";

      int sat(0); //This counter counts the number of recoTracks that are associated to SimTracks from Signal only
      int at(0); //This counter counts the number of recoTracks that are associated to SimTracks
      int rT(0); //This counter counts the number of recoTracks in general
      int seed_fit_failed = 0;
      size_t n_selTrack_dr = 0;

      //calculate dR for tracks
      const edm::View<Track> *trackCollectionDr = &trackCollection;
      if(calculateDrSingleCollection_) {
        trackCollectionDr = trackCollectionForDrCalculation.product();
      }
      declareDynArray(float, trackCollection.size(), dR_trk);
      trackDR(trackCollection, *trackCollectionDr, dR_trk);

      int sumSize = 0, sumCounter = 0, nRecHits = 0;
      int trackAndTp = 0, trackOnly = 0, tpOnly = 0, nloops = 0;

      for(View<Track>::size_type i=0; i<trackCollection.size(); ++i){
        auto track = trackCollection.refAt(i);
	      rT++;
        if(trackFromSeedFitFailed(*track)) ++seed_fit_failed;
        if((*dRTrackSelector)(*track)) ++n_selTrack_dr;

	      bool isSigSimMatched(false);
	      bool isSimMatched(false);
        bool isChargeMatched(true);
        int numAssocRecoTracks = 0;
	      int nSimHits = 0;
	      double sharedFraction = 0.;

        auto tpFound = recSimColl.find(track);
        isSimMatched = tpFound != recSimColl.end();
        if (isSimMatched) {

          const auto& tp = tpFound->val;
          nSimHits = tp[0].first->numberOfTrackerHits();
          sharedFraction = tp[0].second;
            if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
            if(simRecColl.find(tp[0].first) != simRecColl.end()) numAssocRecoTracks = simRecColl[tp[0].first].size();
	             at++;
	    for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite){
              TrackingParticle trackpart = *(tp[tp_ite].first);
	      if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0)){
	      	isSigSimMatched = true;
		sat++;
		break;
	      }
            }
	    LogTrace("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
                                       << " associated with quality:" << tp.begin()->second <<"\n";
	} else {
	  LogTrace("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
                                     << " NOT associated to any TrackingParticle" << "\n";
	}

        // set MVA values for this track
        // take also the indices of first MVAs to select by loose and
        // HP quality
        unsigned int selectsLoose = mvaCollections.size();
        unsigned int selectsHP = mvaCollections.size();
        if(doMVAPlots_) {
          for(size_t imva=0; imva<mvaCollections.size(); ++imva) {
            const auto& mva = *(mvaCollections[imva]);
            const auto& qual = *(qualityMaskCollections[imva]);
            mvaValues.push_back(mva[i]);

            if(selectsLoose >= imva && trackSelected(qual[i], reco::TrackBase::loose))
              selectsLoose = imva;
            if(selectsHP >= imva && trackSelected(qual[i], reco::TrackBase::highPurity))
              selectsHP = imva;
          }
        }

	double dR=dR_trk[i];
	histoProducerAlgo_->fill_generic_recoTrack_histos(w,*track, ttopo, bs.position(), thePVposition, theSimPVPosition, isSimMatched,isSigSimMatched, isChargeMatched, numAssocRecoTracks, puinfo.getPU_NumInteractions(), nSimHits, sharedFraction, dR, mvaValues, selectsLoose, selectsHP);
        mvaValues.clear();

        if(doSummaryPlots_) {
          h_reco_coll[ww]->Fill(www);
          if(isSimMatched) {
            h_assoc2_coll[ww]->Fill(www);
            if(numAssocRecoTracks>1) {
              h_looper_coll[ww]->Fill(www);
            }
            if(!isSigSimMatched) {
              h_pileup_coll[ww]->Fill(www);
            }
          }
        }

	// dE/dx
	if (dodEdxPlots_) histoProducerAlgo_->fill_dedx_recoTrack_histos(w,track, v_dEdx);


	//Fill other histos
	if (!isSimMatched) continue;

	histoProducerAlgo_->fill_simAssociated_recoTrack_histos(w,*track);

	/* TO BE FIXED LATER
	if (associators[ww]=="trackAssociatorByChi2"){
	  //association chi2
	  double assocChi2 = -tp.begin()->second;//in association map is stored -chi2
	  h_assochi2[www]->Fill(assocChi2);
	  h_assochi2_prob[www]->Fill(TMath::Prob((assocChi2)*5,5));
	}
	else if (associators[ww]=="quickTrackAssociatorByHits"){
	  double fraction = tp.begin()->second;
	  h_assocFraction[www]->Fill(fraction);
	  h_assocSharedHit[www]->Fill(fraction*track->numberOfValidHits());
	}
	*/


        if(doResolutionPlots_[www]) {
          //Get tracking particle parameters at point of closest approach to the beamline
          TrackingParticleRef tpr = tpFound->val.begin()->first;
          TrackingParticle::Vector momentumTP = parametersDefinerTP->momentum(event,setup,tpr);
          TrackingParticle::Point vertexTP = parametersDefinerTP->vertex(event,setup,tpr);
          int chargeTP = tpr->charge();

          histoProducerAlgo_->fill_ResoAndPull_recoTrack_histos(w,momentumTP,vertexTP,chargeTP,
                                                                *track,bs.position());
        }


	//TO BE FIXED
	//std::vector<PSimHit> simhits=tpr.get()->trackPSimHit(DetId::Tracker);
	//nrecHit_vs_nsimHit_rec2sim[w]->Fill(track->numberOfValidHits(), (int)(simhits.end()-simhits.begin() ));

      } // End of for(View<Track>::size_type i=0; i<trackCollection.size(); ++i){

      int eveNumber = event.id().event();
      int runNumber = event.id().run();
      int lumNumber = event.id().luminosityBlock();

      for (size_t i = 0; i < allDoublets.size(); ++i)
      {
        auto iHd = allDoublets[i];
        std::string dName = allDoubletsNames[i];

        std::string fileName = std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber);
        fileName += "_" + dName[i] + "_dnn_doublets.txt";
        std::ofstream outCNNFile(fileName, std::ofstream::app);

        for (std::vector<IntermediateHitDoublets::LayerPairHitDoublets>::const_iterator lIt= iHd->layerSetsBegin(); lIt != iHd->layerSetsEnd(); ++lIt)
          {

            std::vector< RecHitsSortedInPhi::Hit> hits;
            std::vector< const SiPixelRecHit*> siHits;

            std::vector< SiPixelRecHit::ClusterRef> clusters;
            std::vector< DetId> detIds;
            std::vector< const GeomDet*> geomDets;
            std::vector <unsigned int> hitIds, subDetIds, detSeqs;

            std::vector< std::vector< float>> hitPars;
            std::vector< float > inHitPars, outHitPars;
            std::vector< float > inTP, outTP, theTP;

            float ax1, ax2, diffADC = 0.0;

            DetLayer const * innerLayer = lIt->doublets().detLayer(HitDoublets::inner);
            if(find(pixelDets.begin(),pixelDets.end(),innerLayer->seqNum())==pixelDets.end()) continue;   //TODO change to std::map ?

            DetLayer const * outerLayer = lIt->doublets().detLayer(HitDoublets::outer);
            if(find(pixelDets.begin(),pixelDets.end(),outerLayer->seqNum())==pixelDets.end()) continue;

            siHits.push_back(dynamic_cast<const SiPixelRecHit*>((hits[0])));
            siHits.push_back(dynamic_cast<const SiPixelRecHit*>((hits[1])));

            for (size_t i = 0; i < lIt->doublets().size(); i++)
            {
              diffADC = 0.0;

              hits.clear(); siHits.clear(); clusters.clear();
              detIds.clear(); geomDets.clear(); hitIds.clear();
              subDetIds.clear(); detSeqs.clear(); hitPars.clear(); theTP.clear();
              inHitPars.clear(); outHitPars.clear();

              for (auto )
              hits.push_back(lIt->doublets().hit(i, HitDoublets::inner)); //TODO CHECK EMPLACEBACK
              hits.push_back(lIt->doublets().hit(i, HitDoublets::outer));

              for (auto h : hits)
              {
                detIds.push_back(h->hit()->geographicalId());
                subDetIds.push_back((h->hit()->geographicalId()).subdetId());
              }
              // innerDetId = innerHit->hit()->geographicalId();

              if (! (((subDetIds[0]==1) || (subDetIds[0]==2)) && ((subDetIds[1]==1) || (subDetIds[1]==2)))) continue;

              siHits.push_back(dynamic_cast<const SiPixelRecHit*>((hits[0])));
              siHits.push_back(dynamic_cast<const SiPixelRecHit*>((hits[1])));

              clusters.push_back(siHits[0]->cluster());
              clusters.push_back(siHits[1]->cluster());

              detSeqs.push_back(innerLayer->seqNum());
              detSeqs.push_back(outerLayer->seqNum());

              geomDets.push_back(hits[0]->det());
              geomDets.push_back(hits[1]->det());

              hitPars.push_back(inHitPars);
              hitPars.push_back(outHitPars);

              HitDoublets::layer layers[2] = {HitDoublets::inner, HitDoublets::outer};

              for(int j = 0; j < 2; ++j)
              {

              //4
                     hitPars[j].push_back((hits[j]->hit()->globalState()).position.x()); //1
                     hitPars[j].push_back((hits[j]->hit()->globalState()).position.y());
                     hitPars[j].push_back((hits[j]->hit()->globalState()).position.z()); //3

                     hitPars[j].push_back(lIt->doublets().phi(i,layers[j])); //Phi //FIXME
                     hitPars[j].push_back(lIt->doublets().r(i,layers[j])); //R //TODO add theta and DR

                     hitPars[j].push_back(detSeqs[j]); //det number //6

                     //Module labels
                     if(subDetIds[j]==1) //barrel
                     {
                       hitPars[j].push_back(float(true)); //isBarrel //7
                       hitPars[j].push_back(PXBDetId(detIds[j]).layer());
                       hitPars[j].push_back(PXBDetId(detIds[j]).ladder());
                       hitPars[j].push_back(-1.0);
                       hitPars[j].push_back(-1.0);
                       hitPars[j].push_back(-1.0);
                       hitPars[j].push_back(PXBDetId(detIds[j]).module()); //14
                     }
                     else
                     {
                       hitPars[j].push_back(float(false)); //isBarrel
                       hitPars[j].push_back(-1.0);
                       hitPars[j].push_back(-1.0);
                       hitPars[j].push_back(PXFDetId(detIds[j]).side());
                       hitPars[j].push_back(PXFDetId(detIds[j]).disk());
                       hitPars[j].push_back(PXFDetId(detIds[j]).panel());
                       hitPars[j].push_back(PXFDetId(detIds[j]).module());
                     }

                     //Module orientation
                     ax1 = geomDets[j]->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp(); //15
                     ax2 = geomDets[j]->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();

                     hitPars[j].push_back(float(ax1<ax2)); //isFlipped
                     hitPars[j].push_back(ax1); //Module orientation y
                     hitPars[j].push_back(ax2); //Module orientation x


                     //TODO check CLusterRef & OmniClusterRef

                     //ClusterInformations
                     hitPars[j].push_back((float)clusters[j]->x()); //20
                     hitPars[j].push_back((float)clusters[j]->y());
                     hitPars[j].push_back((float)clusters[j]->size());
                     hitPars[j].push_back((float)clusters[j]->sizeX());
                     hitPars[j].push_back((float)clusters[j]->sizeY());
                     hitPars[j].push_back((float)clusters[j]->pixel(0).adc); //25
                     hitPars[j].push_back(float(clusters[j]->charge())/float(clusters[j]->size())); //avg pixel charge

                     diffADC -= clusters[j]->charge(); diffADC *= -1.0; //At the end == Outer Hit ADC - Inner Hit ADC

                     hitPars[j].push_back((float)(clusters[j]->sizeX() > padSize));//27
                     hitPars[j].push_back((float)(clusters[j]->sizeY() > padSize));

                     hitPars[j].push_back((float)siHits[j]->spansTwoROCs());
                     hitPars[j].push_back((float)siHits[j]->hasBadPixels());
                     hitPars[j].push_back((float)siHits[j]->isOnEdge()); //31

                     //Cluster Pad
                     TH2F hClust("hClust","hClust",
                                             padSize,
                                             clusters[j]->x()-padHalfSize,
                                             clusters[j]->x()+padHalfSize,
                                             padSize,
                                             clusters[j]->y()-padHalfSize,
                                             clusters[j]->y()+padHalfSize);

                    //Initialization
                     for (int nx = 0; nx < padSize; ++nx)
                      for (int ny = 0; ny < padSize; ++ny)
                       hClust.SetBinContent(nx,ny,0.0);

                     for (int k = 0; k < clusters[j]->size(); ++k)
                       hClust.SetBinContent(hClust.FindBin((float)clusters[j]->pixel(k).x, (float)clusters[j]->pixel(k).y),(float)clusters[j]->pixel(k).adc);

                     //Linearizing the cluster

                     for (int ny = padSize; ny>0; --ny)
                     {
                       for(int nx = 0; nx<padSize; nx++)
                       {
                       int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
                       hitPars[j].push_back(hClust.GetBinContent(n));
                       }
                     }

                     //ADC sum
                     hitPars[j].push_back(float(clusters[j]->charge()));


                   }


              auto rangeIn = tpClust->equal_range(lIt->doublets().hit(i, HitDoublets::inner)->firstClusterRef());
              auto rangeOut = tpClust->equal_range(lIt->doublets().hit(i, HitDoublets::outer)->firstClusterRef());

              std::vector< std::pair<int,int> > kPdgIn, kPdgOut, kIntersection;
              std::vector< int > kIntPdgs;

              for(auto ip=rangeIn.first; ip != rangeIn.second; ++ip)
                kPdgIn.push_back({ip->second.key(),(*ip->second).pdgId()});

              for(auto ip=rangeOut.first; ip != rangeOut.second; ++ip)
                kPdgOut.push_back({ip->second.key(),(*ip->second).pdgId()});

              std::set_intersection(kPdgIn.begin(), kPdgIn.end(),kPdgOut.begin(), kPdgOut.end(), std::back_inserter(kIntersection));

              const TrackingRecHit* inRecHit = dynamic_cast<const TrackingRecHit*> (lIt->doublets().hit(i, HitDoublets::inner));
              const TrackingRecHit* outRecHit = dynamic_cast<const TrackingRecHit*> (lIt->doublets().hit(i, HitDoublets::outer));

              bool trueDoublet = false;
              for (size_t i = 0; i < kIntersection.size(); i++) {
                kIntPdgs.push_back(kIntersection[i].second);
              }
              if(kIntersection.size()>0)
              for(View<Track>::size_type i=0; i<trackCollection.size(); ++i){

                bool inTrue = false, outTrue = false;

                auto track = trackCollection.refAt(i);
        	      rT++;
                if(trackFromSeedFitFailed(*track)) ++seed_fit_failed;
                if((*dRTrackSelector)(*track)) ++n_selTrack_dr;

        	      bool isSigSimMatched(false);
        	      bool isSimMatched(false);
                bool isChargeMatched(true);
                int numAssocRecoTracks = 0;
        	      int nSimHits = 0;
        	      double sharedFraction = 0.;

                auto tpFound = recSimColl.find(track);
                isSimMatched = tpFound != recSimColl.end();
                if (!isSimMatched)
                continue;

                for ( trackingRecHit_iterator recHit = track->recHitsBegin();recHit != track->recHitsEnd(); ++recHit )
                {

                  if(!(*recHit))
                  continue;

                  if (!((*recHit)->isValid()))
                  continue;

                  if(!((*recHit)->hasPositionAndError()))
                  continue;

                  if((*recHit)->sharesInput(inRecHit,TrackingRecHit::SharedInputType::some))
                  {
                    inTrue = true;
                    continue;
                  }

                  if((*recHit)->sharesInput(outRecHit,TrackingRecHit::SharedInputType::some))
                    outTrue = true;
                }

                if(!(outTrue && inTrue))
                  continue;

                const auto& tp = tpFound->val;
                const TrackingParticle& particle = *tp[0].first;

                if(std::find(kIntPdgs.begin(),kIntPdgs.end(),(int)(particle.pdgId())) == kIntPdgs.end())
                  continue;
                else
                  trueDoublet = true;

                nSimHits = tp[0].first->numberOfTrackerHits();
                sharedFraction = tp[0].second;
                if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
                if(simRecColl.find(tp[0].first) != simRecColl.end()) numAssocRecoTracks = simRecColl[tp[0].first].size();
                at++;
                for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite)
                {
                  TrackingParticle trackpart = *(tp[tp_ite].first);
                  if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0))
                  {
                    isSigSimMatched = true;
                    break;
                  }
                }

                TrackingParticle::Vector momTp = particle.momentum();
                TrackingParticle::Point  verTp  = particle.vertex();

                theTP.push_back(1.0); // 1
                theTP.push_back(kPar->second.key()); // 2
                theTP.push_back(momTp.x()); // 3
                theTP.push_back(momTp.y()); // 4
                theTP.push_back(momTp.z()); // 5
                theTP.push_back(particle.pt()); //6

                theTP.push_back(particle.mt());
                theTP.push_back(particle.et());
                theTP.push_back(particle.massSqr()); //9

                theTP.push_back(particle.pdgId());
                theTP.push_back(particle.charge()); //11

                theTP.push_back(particle.numberOfTrackerHits()); //TODO no. pixel hits?
                theTP.push_back(particle.numberOfTrackerLayers());
                //TODO is cosmic?
                theTP.push_back(particle.phi());
                theTP.push_back(particle.eta());
                theTP.push_back(particle.rapidity()); //16

                theTP.push_back(verTp.x());
                theTP.push_back(verTp.y());
                theTP.push_back(verTp.z());
                theTP.push_back((-verTp.x()*sin(momTp.phi())+verTp.y()*cos(momTp.phi()))); //dxy
                theTP.push_back((verTp.z() - (verTp.x() * momTp.x()+
                                  verTp.y() *
                                  momTp.y())/sqrt(momTp.perp2()) *
                                  momTp.z()/sqrt(momTp.perp2())));

                theTP.push_back(particle.eventId().bunchCrossing()); //22
                theTP.push_back(isChargeMatched);
                theTP.push_back(isSigSimMatched);
                theTP.push_back(sharedFraction);
                theTP.push_back(numAssocRecoTracks); //26

                if(trueDoublet)
                  break;
            }
            else
              for (int i = 0; i < tParams; i++)
                theTP.push_back(-1.0);

            outCNNFile << runNumber << "\t" << eveNumber << "\t" << lumNumber << "\t";
            outCNNFile <<innerLayer->seqNum() << "\t" << outerLayer->seqNum() << "\t";
            outCNNFile << bs.x0() << "\t" << bs.y0() << "\t" << bs.z0() << "\t" << bs.sigmaZ() << "\t";
            for (int j = 0; j < 2; j++)
              for (size_t i = 0; i < hitPars[j].size(); i++)
                outCNNFile << hitPars[j][i] << "\t";

            outCNNFile << diffADC << "\t";

            for (size_t i = 0; i < theTP.size(); i++)
              outCNNFile << theTP[i] << "\t";

            outCNNFile << 542.1369;
            outCNNFile << std::endl;

            outCNNFile << hitPars[0].size() << " - " <<hitPars[1].size()<< " - " <<theTP.size() << " - " << std::endl;

          } //hits loop

      } // doublets loop

    } //loop on doublets sets

      mvaCollections.clear();
      qualityMaskCollections.clear();

      histoProducerAlgo_->fill_trackBased_histos(w,at,rT, n_selTrack_dr, n_selTP_dr);
      // Fill seed-specific histograms

      // std::cout << "OVERALL True doublets " << sumCounter << " on "<< sumSize << " with " << nRecHits - 1 << "track doublets" << std::endl;
      // std::cout << "Both : " << trackAndTp << " TrackOnly : "<< trackOnly << " TpOnly"<< nloops > 0 ? tpOnly/nloops : 0  << std::endl;

      if(doSeedPlots_) {
        histoProducerAlgo_->fill_seed_histos(www, seed_fit_failed, trackCollection.size());
      }


      LogTrace("TrackValidator") << "Collection " << www << "\n"
                                 << "Total Simulated (selected): " << n_selTP_dr << "\n"
                                 << "Total Reconstructed (selected): " << n_selTrack_dr << "\n"
                                 << "Total Reconstructed: " << rT << "\n"
                                 << "Total Associated (recoToSim): " << at << "\n"
                                 << "Total Fakes: " << rT-at << "\n";
    } // End of  for (unsigned int www=0;www<label.size();www++){
  } //END of for (unsigned int ww=0;ww<associators.size();ww++){

  // for (size_t j = 0; j < inHitsGP.size(); j++) {
  //   for (size_t i = 0; i < trakHitsGP.size(); i++) {
  //     if(inHitsGP[j]==trakHitsGP[i])
  //     std::cout << "One Match" << std::endl;
  //   }
  // }

}
