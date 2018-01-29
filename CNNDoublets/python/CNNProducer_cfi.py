import FWCore.ParameterSet.Config as cms

CNNDoubletsProducer = cms.EDProducer('CNNDoublets'
        doublets    = cms.InputTag( "initialStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" )
)
