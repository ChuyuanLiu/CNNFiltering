import FWCore.ParameterSet.Config as cms

CNNDoubletsProducer = cms.EDProducer('CNNDoublets'
            src     = cms.InputTag( "initialStepHitDoublets" )ò
)
