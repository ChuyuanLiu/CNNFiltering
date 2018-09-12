import FWCore.ParameterSet.Config as cms
process = cms.Process('test')

import sys

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')

input_file = "QCD_PhiFilter_Dau_HardQCD_25_RECO_RECOSIM_MINIAODSIM.root"

options.register ('file',
				  input_file,
				  VarParsing.multiplicity.singleton,
				  VarParsing.varType.string,
				  "Filename ")

options.parseArguments()

input_file = options.file

process.PsiPhiProducer = cms.EDProducer('TrackAnalyzer',
    PFCandidates        = cms.InputTag('packedPFCandidates')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(input_file)
)

process.dump=cms.EDAnalyzer('EventContentAnalyzer')

process.p = cms.Path(
                     process.tracks *
                     process.dump *
                     )# * process.Phi2KKPAT * process.patSelectedTracks *process.rootupleKK
