import FWCore.ParameterSet.Config as cms
process = cms.Process('test')

import sys

process.PsiPhiProducer = cms.EDProducer('TrackAnalyzer',
    PFCandidates        = cms.InputTag('packedPFCandidates')
)

process.dump=cms.EDAnalyzer('EventContentAnalyzer')

process.p = cms.Path(
                     process.tracks *
                     process.dump *
                     )# * process.Phi2KKPAT * process.patSelectedTracks *process.rootupleKK
