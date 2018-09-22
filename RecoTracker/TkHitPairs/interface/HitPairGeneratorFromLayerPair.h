#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPair {

public:

  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;

  HitPairGeneratorFromLayerPair(unsigned int inner,
                                unsigned int outer,
                                LayerCacheType* layerCache,
				unsigned int max=0);

  ~HitPairGeneratorFromLayerPair();

  HitDoublets doublets( const TrackingRegion& reg,
                        const edm::Event & ev,  const edm::EventSetup& es, Layers layers) {
    assert(theLayerCache);
    return doublets(reg, ev, es, layers, *theLayerCache);
  }
  HitDoublets doublets( const TrackingRegion& reg,
                        const edm::Event & ev,  const edm::EventSetup& es, const Layer& innerLayer, const Layer& outerLayer) {
    assert(theLayerCache);
    return doublets(reg, ev, es, innerLayer, outerLayer, *theLayerCache);
  }
  HitDoublets doublets( const TrackingRegion& reg,
                        const edm::Event & ev, const edm::EventSetup& es, Layers layers, LayerCacheType& layerCache) {
    Layer innerLayerObj = innerLayer(layers);
    Layer outerLayerObj = outerLayer(layers);
    return doublets(reg, ev, es, innerLayerObj, outerLayerObj, layerCache);
  }
  HitDoublets doublets( const TrackingRegion& reg,
                        const edm::Event & ev,  const edm::EventSetup& es, const Layer& innerLayer, const Layer& outerLayer, LayerCacheType& layerCache);

  void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,
                 const edm::Event & ev,  const edm::EventSetup& es, Layers layers);
  static void doublets(
						      const TrackingRegion& region,
						      const DetLayer & innerHitDetLayer,
						      const DetLayer & outerHitDetLayer,
						      const RecHitsSortedInPhi & innerHitsMap,
						      const RecHitsSortedInPhi & outerHitsMap,
						      const edm::EventSetup& iSetup,
						      const unsigned int theMaxElement,
						      HitDoublets & result);

  bool makeInference(const BaseTrackerRecHit* innerHit,
                     const BaseTrackerRecHit* outerHit,
                     tensorflow::Session* session,
                     int inSeq, int outSeq,
                     int inLay, int outLay,
                     float t);

  Layer innerLayer(const Layers& layers) const { return layers[theInnerLayer]; }
  Layer outerLayer(const Layers& layers) const { return layers[theOuterLayer]; }

private:
  LayerCacheType *theLayerCache;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
};

#endif
