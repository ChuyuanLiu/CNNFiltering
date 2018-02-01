import FWCore.ParameterSet.Config as cms

# ++IntermediateHitDoublets "detachedTripletStepHitDoublets" "" "RECO" (productId = 3:311)
# ++IntermediateHitDoublets "initialStepHitDoublets" "" "RECO" (productId = 3:312)
# ++IntermediateHitDoublets "initialStepHitDoubletsPreSplitting" "" "RECO" (productId = 3:313)
# ++IntermediateHitDoublets "lowPtTripletStepHitDoublets" "" "RECO" (productId = 3:314)
# ++IntermediateHitDoublets "mixedTripletStepHitDoubletsA" "" "RECO" (productId = 3:315)
# ++IntermediateHitDoublets "mixedTripletStepHitDoubletsB" "" "RECO" (productId = 3:316)
# ++IntermediateHitDoublets "pixelLessStepHitDoublets" "" "RECO" (productId = 3:317)
# ++IntermediateHitDoublets "tobTecStepHitDoubletsTripl" "" "RECO" (productId = 3:318)
# ++IntermediateHitDoublets "tripletElectronHitDoublets" "" "RECO" (productId = 3:319)

CNNDetTriplet = cms.EDProducer('CNNAnalyze',
        doublets    = cms.InputTag( "initialStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" )
)




CNNDoubletsSequence = cms.Sequence(PsiPhiProducer*PsiPhiFitter*rootuple)
