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

# 2018
# ++IntermediateHitDoublets "detachedQuadStepHitDoublets" "" "RECO" (productId = 3:309)
# ++IntermediateHitDoublets "detachedTripletStepHitDoublets" "" "RECO" (productId = 3:310)
# ++IntermediateHitDoublets "highPtTripletStepHitDoublets" "" "RECO" (productId = 3:311)
# ++IntermediateHitDoublets "initialStepHitDoublets" "" "RECO" (productId = 3:312)
# ++IntermediateHitDoublets "initialStepHitDoubletsPreSplitting" "" "RECO" (productId = 3:313)
# ++IntermediateHitDoublets "lowPtQuadStepHitDoublets" "" "RECO" (productId = 3:314)
# ++IntermediateHitDoublets "lowPtTripletStepHitDoublets" "" "RECO" (productId = 3:315)
# ++IntermediateHitDoublets "mixedTripletStepHitDoubletsA" "" "RECO" (productId = 3:316)
# ++IntermediateHitDoublets "mixedTripletStepHitDoubletsB" "" "RECO" (productId = 3:317)
# ++IntermediateHitDoublets "pixelLessStepHitDoublets" "" "RECO" (productId = 3:318)
# ++IntermediateHitDoublets "tobTecStepHitDoubletsTripl" "" "RECO" (productId = 3:319)
# ++IntermediateHitDoublets "tripletElectronHitDoublets" "" "RECO" (productId = 3:320)

detachedQuadStepCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "detachedQuadStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

detachedTripletStepCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "detachedTripletStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

initialStepPreSplittingCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "initialStepHitDoubletsPreSplitting" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

lowPtQuadStepCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "lowPtQuadStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

mixedTripletStepACNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "mixedTripletStepHitDoubletsA" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

mixedTripletStepBCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "mixedTripletStepHitDoubletsB" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

pixelLessStepCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "pixelLessStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

tobTecStepCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "tobTecStepHitDoubletsTripl" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

tripletElectronCNN = cms.EDAnalyzer('CNNAnalyze',
        doublets    = cms.InputTag( "tripletElectronHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)



CNNDoubletsSequence = cms.Sequence(detachedQuadStepCNN *
                                   detachedTripletStepCNN *
                                   initialStepPreSplittingCNN *
                                   lowPtQuadStepCNN *
                                   mixedTripletStepACNN *
                                   mixedTripletStepBCNN *
                                   pixelLessStepCNN *
                                   tobTecStepCNN *
                                   tripletElectronCNN)
