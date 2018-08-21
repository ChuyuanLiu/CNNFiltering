import FWCore.ParameterSet.Config as cms
process = cms.Process('phikkml')

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')

options.register ('i',
				  0,
				  VarParsing.multiplicity.singleton,
				  VarParsing.varType.int,
				  "Sequential number ")

options.parseArguments()

input_file = "file:/lustre/cms/store/user/adiflori/GEN-MINIAODSIMBBbar_JpsiFilter_HardQCD_50/crab_GEN-MINIAODSIM_BBbar_JpsiFilter_HardQCD_50_20180805_104626/180805_084640/0000/BBbar_JpsiFilter_HardQCD_MINIAODSIM_PU40_100.root"

process.load("CNNFiltering.CNNAnalyze.CNNTracksAnalyze_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_dataRun2_ReReco_EOY17_v1')
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_dataRun2_ReReco_EOY17_v2') #F
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_mc2017_realistic_v11')
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2018_realistic_v10')

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(input_file)
)

# process.TFileService = cms.Service("TFileService",
#         fileName = cms.string('rootuple-2017-dimuonditrak.root'),
# )

kaonmass = 0.493677

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

charmoniumHLT = [
#Phi
'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05',
#JPsi
#'HLT_DoubleMu4_JpsiTrkTrk_Displaced',
#'HLT_DoubleMu4_JpsiTrk_Displaced',
#'HLT_DoubleMu4_Jpsi_Displaced',
#'HLT_DoubleMu4_3_Jpsi_Displaced',
#'HLT_Dimuon20_Jpsi_Barrel_Seagulls',
#'HLT_Dimuon25_Jpsi',
]

hltList = charmoniumHLT #muoniaHLT

hltpaths = cms.vstring(hltList)

hltpathsV = cms.vstring([h + '_v*' for h in hltList])

filters = cms.vstring(
                                #HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi
                                #'hltDoubleMu2JpsiDoubleTrkL3Filtered',
                                'hltDoubleTrkmumuFilterDoubleMu2Jpsi',
								'hltJpsiTkTkVertexFilterPhiDoubleTrk1v1',
                                'hltJpsiTkTkVertexFilterPhiDoubleTrk1v2',
								'hltJpsiTkTkVertexFilterPhiDoubleTrk1v3',
								'hltJpsiTkTkVertexFilterPhiDoubleTrk1v4',
								'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v1',
								'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v2',
								'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v3',
								'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v4',
                                #HLT_DoubleMu4_JpsiTrkTrk_Displaced_v4
                                # 'hltDoubleMu4JpsiDisplacedL3Filtered'
                                # 'hltDisplacedmumuFilterDoubleMu4Jpsi',
                                # 'hltJpsiTkTkVertexFilterPhiKstar',
                                # #HLT_DoubleMu4_JpsiTrk_Displaced_v12
                                # #'hltDoubleMu4JpsiDisplacedL3Filtered',
                                # 'hltDisplacedmumuFilterDoubleMu4Jpsi',
                                # #'hltJpsiTkVertexProducer',
                                # #'hltJpsiTkVertexFilter',
                                # #HLT_DoubleMu4_Jpsi_Displaced
                                # #'hltDoubleMu4JpsiDisplacedL3Filtered',
                                # #'hltDisplacedmumuVtxProducerDoubleMu4Jpsi',
                                # 'hltDisplacedmumuFilterDoubleMu4Jpsi',
                                # #HLT_DoubleMu4_3_Jpsi_Displaced
                                # #'hltDoubleMu43JpsiDisplacedL3Filtered',
                                # 'hltDisplacedmumuFilterDoubleMu43Jpsi',
                                # #HLT_Dimuon20_Jpsi_Barrel_Seagulls
                                # #'hltDimuon20JpsiBarrelnoCowL3Filtered',
                                # 'hltDisplacedmumuFilterDimuon20JpsiBarrelnoCow',
                                # #HLT_Dimuon25_Jpsi
                                # 'hltDisplacedmumuFilterDimuon25Jpsis'
                                )

process.triggerSelection = cms.EDFilter("TriggerResultsFilter",
                                        triggerConditions = cms.vstring(hltpathsV),
                                        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                        l1tResults = cms.InputTag( "" ),
                                        throw = cms.bool(False)
                                        )

# process.unpackPatTriggers = cms.EDProducer("PATTriggerObjectStandAloneUnpacker",
#   patTriggerObjectsStandAlone = cms.InputTag( 'slimmedPatTrigger' ), #selectedPatTrigger for MC
#   triggerResults              = cms.InputTag( 'TriggerResults::HLT' ),
#   unpackFilterLabels          = cms.bool( True )
# )


process.tracksCNN = cms.EDAnalyzer('CNNTrackDump',
        seqNumber       = cms.int32(options.i),
        processName     = cms.string( "generalTracksCNN"),
        tracks          = cms.InputTag( "generalTracks" ),
        tpMap           = cms.InputTag( "tpClusterProducer" ),
        trMap           = cms.InputTag("trackingParticleRecoTrackAsssociation"),
        genMap          = cms.InputTag("TrackAssociatorByChi2"),
        genParticles    = cms.InputTag("genParticles"),
        traParticles    = cms.InputTag("mix","MergedTrackTruth"),
        beamSpot        = cms.InputTag("offlineBeamSpot"),
        infoPileUp      = cms.InputTag("addPileupInfo")
)

# process.kaonTracks = cms.EDProducer("MCTruthDeltaRMatcher",
#     pdgId = cms.vint32(321),
#     src = cms.InputTag("generalTracks"),
#     distMin = cms.double(0.25),
#     matched = cms.InputTag("genParticleCandidates")
# )
#
# process.kaonsCNN = cms.EDAnalyzer('CNNTrackDump',
#         processName     = cms.string( "kaonTracksCNN"),
#         tracks          = cms.InputTag( "generalTracks" ),
#         tpMap           = cms.InputTag( "tpClusterProducer" ),
#         trMap           = cms.InputTag("trackingParticleRecoTrackAsssociation"),
#         genMap          = cms.InputTag("TrackAssociatorByChi2"),
#         genParticles    = cms.InputTag("genParticles"),
#         traParticles    = cms.InputTag("mix","MergedTrackTruth"),
#         beamSpot        = cms.InputTag("offlineBeamSpot"),
#         infoPileUp      = cms.InputTag("addPileupInfo")
# )

process.phitokk = cms.EDAnalyzer('DiTrack',
         seqNumber          = cms.int32(options.i),
         tracks             = cms.InputTag( "generalTracks"),
         TrakTrakMassCuts   = cms.vdouble(1.0,1.04),
         MassTraks          = cms.vdouble(kaonmass,kaonmass)
         )


process.p = cms.Path(process.triggerSelection * process.tracksCNN * process.phitokk)

#CNNTrackSequence = cms.Sequence(tracksCNN)
