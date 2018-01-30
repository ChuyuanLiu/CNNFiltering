import FWCore.ParameterSet.Config as cms

CNNDoubletsAnalyzer = cms.EDProducer('CNNAnalyze',
        doublets    = cms.InputTag( "initialStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" )
)
