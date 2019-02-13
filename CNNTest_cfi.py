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
#pixelPairStepHitDoublets
#jetCoreRegionalStepHitDoublets

pixelPairStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "pixelPairStepHitDoublets"),
        doublets    = cms.InputTag( "pixelPairStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

jetCoreRegionalStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "pixelPairStepHitDoublets"),
        doublets    = cms.InputTag( "pixelPairStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

detachedQuadStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "detachedQuadStepHitDoublets"),
        doublets    = cms.InputTag( "detachedQuadStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

highPtTripletStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "highPtTripletStepHitDoublets"),
        doublets    = cms.InputTag( "highPtTripletStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

detachedTripletStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "detachedTripletStepHitDoublets" ),
        doublets    = cms.InputTag( "detachedTripletStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

initialStepPreSplittingCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "initialStepHitDoubletsPreSplitting"),
        doublets    = cms.InputTag( "initialStepHitDoubletsPreSplitting" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

initialStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "initialStepHitDoublets"),
        doublets    = cms.InputTag( "initialStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

lowPtQuadStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "lowPtQuadStepHitDoublets" ),
        doublets    = cms.InputTag( "lowPtQuadStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

lowPtTripletStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "lowPtTripletStepHitDoublets" ),
        doublets    = cms.InputTag( "lowPtTripletStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

mixedTripletStepACNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "mixedTripletStepHitDoubletsA"),
        doublets    = cms.InputTag( "mixedTripletStepHitDoubletsA" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

mixedTripletStepBCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "mixedTripletStepHitDoubletsB" ),
        doublets    = cms.InputTag( "mixedTripletStepHitDoubletsB" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

pixelLessStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "pixelLessStepHitDoublets" ),
        doublets    = cms.InputTag( "pixelLessStepHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

tobTecStepCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "tobTecStepHitDoubletsTripl" ),
        doublets    = cms.InputTag( "tobTecStepHitDoubletsTripl" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)

tripletElectronCNN = cms.EDAnalyzer('CNN_TF_Test',
        processName = cms.string( "tripletElectronHitDoublets" ),
        doublets    = cms.InputTag( "tripletElectronHitDoublets" ),
        tpMap       = cms.InputTag( "tpClusterProducer" ),
        beamSpot    = cms.InputTag("offlineBeamSpot"),
        infoPileUp  = cms.InputTag("addPileupInfo")
)



CNNTestSequence = cms.Sequence(detachedQuadStepCNN *
                                   highPtTripletStepCNN *
                                   detachedTripletStepCNN *
                                   initialStepPreSplittingCNN *
                                   initialStepCNN *
                                   lowPtQuadStepCNN *
                                   lowPtTripletStepCNN *
                                   mixedTripletStepACNN *
                                   mixedTripletStepBCNN *
                                   tripletElectronCNN )
                                   #pixelPairStepCNN *
                                   #jetCoreRegionalStepCNN)
