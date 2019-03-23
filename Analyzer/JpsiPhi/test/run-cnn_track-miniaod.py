import FWCore.ParameterSet.Config as cms
process = cms.Process('cnnTracks')

from FWCore.ParameterSet.VarParsing import VarParsing

import sys
sys.path.append("mclists/")

par = VarParsing ('analysis')

par.register ('gtag',
                                  "101X_dataRun2_Prompt_v11",
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.string,
                                  "Global Tag")
par.register ('mc',
                                  "y4506_lhcb",
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.string,
                                  "MC Dataset")

par.register ('filein',
                                  "file:1401AF4A-447C-E811-8EEB-FA163E35DF95.root",
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.string,
                                  "Inputfile")

par.register ('dataset',
                                  "QCD_Soft",
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.string,
                                  "Dataset")

par.register ('ss',
                                  True,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Do Same Sign")

par.register ('isMC',
                                  False,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Is MC?")

par.register ('isLocal',
                                  False,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Is local?")

par.register ('isDebug',
                                  False,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Debug for you,sir?")

par.register ('isY',
                                  False,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Y for you,sir?")

par.register ('kMass',
                                  0.493677,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "KMass")

par.register ('isGen',
                                  False,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.bool,
                                  "Is gen counting only?")

par.register ('i',
                                  0,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.int,
                                  "i")

par.register ('n',
                                  50000,
                                  VarParsing.multiplicity.singleton,
                                  VarParsing.varType.int,
                                  "n")

par.parseArguments()

i = par.i
ismc = par.isMC
doss = par.ss

gen_file = "file:32B83273-030F-E811-9105-E0071B7AF7C0.root"
input_file = "file:006425F0-6DED-E711-850C-0025904C66E8.root"
mc_file = "file:py8_JPsiMM_EvtGen_13TeV_TuneCP5_cfi.root"
mc_file = "file:02CA3723-CEF3-E711-B1CC-4C79BA1810EF.root"
mc_file = "file:FCD01A2E-A6F5-E711-ACA1-003048F5ADF6.root"
runb2018 = "file:1401AF4A-447C-E811-8EEB-FA163E35DF95.root"
input_file = par.filein #runb2018 #gen_file

filename = "data" + par.dataset

from all2018Hlts import *

'''
if par.isLocal:

    from bujpsiphi_filelist import *
    from bsjpsiphi_filelist import *
    from bbbar_filelist import *
    from bbbar_soft_filelist import *
    from y4140_lhcb_filelist import *
    from y4140_zero_filelist import *
    from y4273_lhcb_filelist import *
    from y4273_zero_filelist import *
    from y4704_zero_filelist import *
    from y4704_lhcb_filelist import *
    from y4506_lhcb_filelist import *
    from y4506_zero_filelist import *
    #from y4273_spin_filelist import *
    #from y4506_spin_filelist import *
    from qcd_ml_filelist import *
    from bbbar_ml_filelist import *
    from bstojpsiphi_softqcd_filelist import *
    from bbhook_filelist import *
    from BBbar_Hook_Samet import *

    filename = par.mc

    fileLists = {"qcd_ml" : qcd_ml_filelist,"bbbar_hard" : bbbar_file_list, "bbbar_soft" : bbbar_soft_filelist,
		 "bbhook_samet": BBbar_Hook_Samet,
                 "y4273_zero" : y4273_zero_filelist, "y4273_lhcb" : y4273_lhcb_filelist ,
                 "y4140_lhcb" : y4140_lhcb_filelist, "y4140_zero" : y4140_zero_filelist,
                 "y4506_lhcb" : y4506_lhcb_filelist, "y4506_zero" : y4506_zero_filelist,
                 "y4704_lhcb" : y4704_lhcb_filelist, "y4704_zero" : y4704_zero_filelist,
                # "y4273_spin" : y4273_spin_filelist, "y4506_spin" : y4506_spin_filelist,
                 "bstojpsiphi_softqcd" : bstojpsiphi_softqcd_file_list, "bsjpsiphi" : bsjpsiphi_filelist,
                 "bujpsiphi" : bujpsiphi_filelist, "bbbar_hook": bbhook_filelist}

    gtags = {"qcd_ml" : "100X_upgrade2018_realistic_v10", "bbhook_samet" : "100X_upgrade2018_realistic_v10", "bbbar_hard" : "100X_upgrade2018_realistic_v10",
                 "bbbar_soft" : "100X_upgrade2018_realistic_v10", "bbbar_hook" : "100X_upgrade2018_realistic_v10",
                 "y4273_zero" : "100X_upgrade2018_realistic_v10",  "y4273_lhcb" : "100X_upgrade2018_realistic_v10"  ,
                 "y4140_lhcb" : "100X_upgrade2018_realistic_v10",  "y4140_zero" : "100X_upgrade2018_realistic_v10",
                 "y4506_lhcb" : "100X_upgrade2018_realistic_v10",  "y4506_zero" : "100X_upgrade2018_realistic_v10",
                 "y4704_lhcb" : "100X_upgrade2018_realistic_v10",  "y4704_zero" : "100X_upgrade2018_realistic_v10",
                 "y4273_spin" : "100X_upgrade2018_realistic_v10",  "y4506_spin" : "100X_upgrade2018_realistic_v10",
                 "bujpsiphi" : "100X_upgrade2018_realistic_v10", "bsjpsiphi" : "100X_upgrade2018_realistic_v10",
                 "bstojpsiphi_softqcd" : "94X_mc2017_realistic_v10" }

    par.gtag = gtags[filename]
    n= par.n

    filelist = fileLists[filename] #bbbar_soft_list#bbbar_file_list
    size = (len(filelist) + n) / n
    input_file = filelist[min(i*size,len(filelist)):min((i+1)*size,len(filelist))]
    print min((i+1)*size,len(filelist))
'''

if par.isGen:
    filename = filename + "_genOnly_"

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_dataRun2_ReReco_EOY17_v1')
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_dataRun2_ReReco_EOY17_v2') #F
#process.GlobalTag = GlobalTag(process.GlobalTag, '94X_mc2017_realistic_v11')
#process.GlobalTag = GlobalTag(process.GlobalTag, par.gtag)
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 5

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(input_file),
    #eventsToProcess = cms.untracked.VEventRange(eventsToProcess),
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(250))

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trakAnalyse_'+ filename  + '_' + str(i) +'.root'),
)

kaonmass = 0.493677
pionmass = 0.13957061


charmoniumHLT = [
#Phi
'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi', #2017
'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05', #2018
'HLT_Dimuon25_Jpsi'
#JPsi
#'HLT_DoubleMu4_JpsiTrkTrk_Displaced',
#'HLT_DoubleMu4_JpsiTrk_Displaced',
#'HLT_DoubleMu4_Jpsi_Displaced',
#'HLT_DoubleMu4_3_Jpsi_Displaced',
#'HLT_Dimuon20_Jpsi_Barrel_Seagulls',

]

year = "2018"

if "2017" in par.dataset:
    year = "2017"
if "2016" in par.dataset:
    year = "2016"

hlts = {}

hlts["2018"] = all2018Hlts

hlts["2017"] =["HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi",
    "HLT_Dimuon25_Jpsi",
    "HLT_Dimuon20_Jpsi_Barrel_Seagulls",
    "HLT_DoubleMu4_JpsiTrk_Displaced",
    "HLT_DoubleMu4_JpsiTrkTrk_Displaced"]

hlts["2016"]=["HLT_Dimuon20_Jpsi",
    "HLT_Dimuon16_Jpsi",
    "HLT_Dimuon10_Jpsi_Barrel",
    "HLT_DoubleMu4_JpsiTrk_Displaced"]

filts = {}

filts["2018"] = ['hltDoubleMu2JpsiDoubleTrkL3Filtered',
'hltDoubleTrkmumuFilterDoubleMu2Jpsi',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v1',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v2',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v3',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v4',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v5',
'hltJpsiTkTkVertexFilterPhiDoubleTrk1v6',

'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v1',
'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v2',
'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v3',
'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v4',
'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v5',
'hltJpsiTrkTrkVertexProducerPhiDoubleTrk1v6',

'hltDimuon25JpsiL3fL3Filtered'
]

filts["2017"] =["hltDimuon25JpsiL3fL3Filtered",
"hltDimuon20JpsiBarrelnoCowL3Filtered",
"hltDisplacedmumuFilterDoubleMu4Jpsi",
"hltDoubleMu4JpsiDisplacedL3Filtered",
"hltJpsiTkTkVertexFilterPhiKstar"]

filts["2016"] =["hltDimuon20JpsiL3Filtered",
"hltDimuon16JpsiL3Filtered",
"hltDisplacedmumuFilterDimuon10JpsiBarrel",
"hltDisplacedmumuFilterDoubleMu4Jpsi",
"hltDoubleMu4JpsiDisplacedL3Filtered",
"hltJpsiTkVertexFilter"]


hltpaths = cms.vstring(hlts[year] )
hltpathsV = cms.vstring([h + '_v*' for h in hlts[year] ])

filters = cms.vstring(filts[year])

process.triggerSelection = cms.EDFilter("TriggerResultsFilter",
                                        triggerConditions = cms.vstring(hltpathsV),
                                        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                        l1tResults = cms.InputTag( "" ),
                                        throw = cms.bool(False)
                                        )

process.muonMatch = cms.EDProducer("MCMatcher", # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src     = cms.InputTag("slimmedMuons"), # RECO objects to match
    matched = cms.InputTag("prunedGenParticles"),   # mc-truth particle collection
    mcPdgId     = cms.vint32(13), # one or more PDG ID (13 = muon); absolute values (see below)
    checkCharge = cms.bool(True), # True = require RECO and MC objects to have the same charge
    mcStatus = cms.vint32(1,3,91),     # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR = cms.double(0.5),  # Minimum deltaR for the match
    maxDPtRel = cms.double(0.75),  # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True), # False = just match input in order; True = pick lowest deltaR pair first
)

process.trackMatch = cms.EDProducer("MCMatcher", # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src     = cms.InputTag("packedPFCandidates"), # RECO objects to match
    matched = cms.InputTag("prunedGenParticles"),   # mc-truth particle collection
    mcPdgId     = cms.vint32(321,211,13,2212), # one or more PDG ID (13 = muon); absolute values (see below)
    checkCharge = cms.bool(True), # True = require RECO and MC objects to have the same charge
    mcStatus = cms.vint32(1,3,91,2),     # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR = cms.double(0.5),  # Minimum deltaR for the match
    maxDPtRel = cms.double(0.75),  # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True), # False = just match input in order; True = pick lowest deltaR pair first
)

process.unpackPatTriggers = cms.EDProducer("PATTriggerObjectStandAloneUnpacker",
  patTriggerObjectsStandAlone = cms.InputTag( 'slimmedPatTrigger' ), #selectedPatTrigger for MC
  triggerResults              = cms.InputTag( 'TriggerResults::HLT' ),
  unpackFilterLabels          = cms.bool( True )
)

process.TrackProducer   = cms.EDProducer('CNNTracks',
    PFCandidates        = cms.InputTag('packedPFCandidates'),
    muons               = cms.InputTag("slimmedMuons"),
    TrackMatcher        = cms.InputTag("trackMatch"),
    HLTs                = hltpaths,
    TriggerResults      = cms.InputTag( "TriggerResults", "", "HLT" ),
    primaryVertexTag    = cms.InputTag("offlinePrimaryVertices")
)

process.dump=cms.EDAnalyzer('EventContentAnalyzer')

#triggering = process.triggerSelection

allsteps = process.trackMatch * process.TrackProducer

if par.isDebug:
    allsteps = allsteps * process.dump
process.p = cms.Path(allsteps)
