#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"

using namespace GeomDetEnumerators;
using namespace std;

typedef PixelRecoRange<float> Range;

namespace {
  template<class T> inline T sqr( T t) {return t*t;}
}


#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(
							     unsigned int inner,
							     unsigned int outer,
							     LayerCacheType* layerCache,
							     unsigned int max)
  : theLayerCache(layerCache), theOuterLayer(outer), theInnerLayer(inner), theMaxElement(max)
{
}

HitPairGeneratorFromLayerPair::~HitPairGeneratorFromLayerPair() {}

// devirtualizer
#include<tuple>
namespace {

  template<typename Algo>
  struct Kernel {
    using  Base = HitRZCompatibility;
    void set(Base const * a) {
      assert( a->algo()==Algo::me);
      checkRZ=reinterpret_cast<Algo const *>(a);
    }

    void operator()(int b, int e, const RecHitsSortedInPhi & innerHitsMap, bool * ok) const {
      constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f);
      for (int i=b; i!=e; ++i) {
	Range allowed = checkRZ->range(innerHitsMap.u[i]);
	float vErr = nSigmaRZ * innerHitsMap.dv[i];
	Range hitRZ(innerHitsMap.v[i]-vErr, innerHitsMap.v[i]+vErr);
	Range crossRange = allowed.intersection(hitRZ);
	ok[i-b] = ! crossRange.empty() ;
      }
    }
    Algo const * checkRZ;

  };


  template<typename ... Args> using Kernels = std::tuple<Kernel<Args>...>;

}


bool makeInference(BaseTrackerRecHit const *innerHit,
                   BaseTrackerRecHit const *outerHit,
                   tensorflow::Session* session,
                   int inSeq, int outSeq,
                   int inLay, int outLay,
                   float t)
{

  int numOfDoublets = 1, padSize = 16, cnnLayers = 10, infoSize = 67;
  float padHalfSize = 8.0;

  tensorflow::Tensor inputPads(tensorflow::DT_FLOAT, {numOfDoublets,padSize,padSize,cnnLayers*2});
  tensorflow::Tensor inputFeat(tensorflow::DT_FLOAT, {numOfDoublets,infoSize});

  float* vPad = inputPads.flat<float>().data();
  float* vLab = inputFeat.flat<float>().data();

  std::vector <int> detSeqs, layerIds, subDetIds, detIds;

  detSeqs.push_back(inSeq);
  detSeqs.push_back(outSeq);

  layerIds.push_back(inLay);
  layerIds.push_back(outLay);

  float deltaA = 0.0, deltaADC = 0.0, deltaS = 0.0, deltaR = 0.0;
  float deltaPhi = 0.0, deltaZ = 0.0, zZero = 0.0;

  int iD = 0;
  int iLab = 0;
  int doubOffset = (padSize*padSize*cnnLayers*2)*iD, infoOffset = (infoSize)*iD;

  std::vector< const SiPixelRecHit*> siHits;

  siHits.push_back(dynamic_cast<const SiPixelRecHit*>(innerHit));
  siHits.push_back(dynamic_cast<const SiPixelRecHit*>(outerHit));

  detIds.push_back(innerHit->geographicalId());
  subDetIds.push_back((innerHit->geographicalId()).subdetId());

  detIds.push_back(outerHit->geographicalId());
  subDetIds.push_back((outerHit->geographicalId()).subdetId());

  if (! (((subDetIds[0]==1) || (subDetIds[0]==2)) && ((subDetIds[1]==1) || (subDetIds[1]==2)))) return true;
  //
  // hitPads.push_back(inPad);
  // hitPads.push_back(outPad);

  for(int j = 0; j < 2; ++j)
  {

    int padOffset = layerIds[j] * padSize * padSize + j * padSize * padSize * cnnLayers;

    vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.x(); iLab++;
    vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.y(); iLab++;
    vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.z(); iLab++;

    float p = (float)(siHits[j]->globalPosition().barePhi());
    float phi = p >=0.0 ? p : 2*M_PI + p;
    vLab[iLab + infoOffset] = phi; iLab++;
    float r = (float)(siHits[j]->globalPosition().perp());
    vLab[iLab + infoOffset] = r;//(float)(siHits[j]->globalPosition().perp());//(float)copyDoublets.r(iD,layers[j]); iLab++;

    vLab[iLab + infoOffset] = (float)detSeqs[j]; iLab++;

    if(subDetIds[j]==1) //barrel
    {

      vLab[iLab + infoOffset] = float(true); iLab++; //isBarrel //7
      vLab[iLab + infoOffset] = PXBDetId(detIds[j]).layer(); iLab++;
      vLab[iLab + infoOffset] = PXBDetId(detIds[j]).ladder(); iLab++;
      vLab[iLab + infoOffset] = -1.0; iLab++;
      vLab[iLab + infoOffset] = -1.0; iLab++;
      vLab[iLab + infoOffset] = -1.0; iLab++;
      vLab[iLab + infoOffset] = PXBDetId(detIds[j]).module(); iLab++; //14

    }
    else
    {
      vLab[iLab + infoOffset] = float(false); iLab++; //isBarrel
      vLab[iLab + infoOffset] = -1.0; iLab++;
      vLab[iLab + infoOffset] = -1.0; iLab++;
      vLab[iLab + infoOffset] = PXFDetId(detIds[j]).side(); iLab++;
      vLab[iLab + infoOffset] = PXFDetId(detIds[j]).disk(); iLab++;
      vLab[iLab + infoOffset] = PXFDetId(detIds[j]).panel(); iLab++;
      vLab[iLab + infoOffset] = PXFDetId(detIds[j]).module(); iLab++;
    }

    //Module orientation
    float ax1  = siHits[j]->det()->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp(); //15
    float ax2  = siHits[j]->det()->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();

    vLab[iLab + infoOffset] = float(ax1<ax2); iLab++; //isFlipped
    vLab[iLab + infoOffset] = ax1; iLab++; //Module orientation y
    vLab[iLab + infoOffset] = ax2; iLab++; //Module orientation x

    auto thisCluster = siHits[j]->cluster();
    //TODO check CLusterRef & OmniClusterRef

    float xC = (float) thisCluster->x(), yC = (float) thisCluster->y();

//inX
    vLab[iLab + infoOffset] = (float)xC; iLab++; //20
    vLab[iLab + infoOffset] = (float)yC; iLab++;
    vLab[iLab + infoOffset] = (float)thisCluster->size(); iLab++;
    vLab[iLab + infoOffset] = (float)thisCluster->sizeX(); iLab++;
    vLab[iLab + infoOffset] = (float)thisCluster->sizeY(); iLab++;
    vLab[iLab + infoOffset] = (float)thisCluster->pixel(0).adc; iLab++; //25
    vLab[iLab + infoOffset] = float(thisCluster->charge())/float(thisCluster->size()); iLab++; //avg pixel charge

    vLab[iLab + infoOffset] = (float)(thisCluster->sizeX() > padSize); iLab++;//27
    vLab[iLab + infoOffset] = (float)(thisCluster->sizeY() > padSize); iLab++;
    vLab[iLab + infoOffset] = (float)(thisCluster->sizeY()) / (float)(thisCluster->sizeX()); iLab++;

    vLab[iLab + infoOffset] = (float)siHits[j]->spansTwoROCs(); iLab++;
    vLab[iLab + infoOffset] = (float)siHits[j]->hasBadPixels(); iLab++;
    vLab[iLab + infoOffset] = (float)siHits[j]->isOnEdge(); iLab++; //31

    vLab[iLab + infoOffset] = (float)(thisCluster->charge()); iLab++;

    deltaA   -= ((float)thisCluster->size()); deltaA *= -1.0;
    deltaADC -= thisCluster->charge(); deltaADC *= -1.0; //At the end == Outer Hit ADC - Inner Hit ADC
    deltaS   -= ((float)(thisCluster->sizeY()) / (float)(thisCluster->sizeX())); deltaS *= -1.0;
    deltaR   -= r; deltaR *= -1.0; //copyDoublets.r(iD,layers[j])
    deltaPhi -= phi; deltaPhi *= -1.0;

    // TH2F hClust("hClust","hClust",
    // padSize,
    // thisCluster->x()-padSize/2,
    // thisCluster->x()+padSize/2,
    // padSize,
    // thisCluster->y()-padSize/2,
    // thisCluster->y()+padSize/2);
    //
    // //Initialization
    // for (int nx = 0; nx < padSize; ++nx)
    // for (int ny = 0; ny < padSize; ++ny)
    // hClust.SetBinContent(nx,ny,0.0);
    //
    // for (int k = 0; k < thisCluster->size(); ++k)
    // hClust.SetBinContent(hClust.FindBin((float)thisCluster->pixel(k).x, (float)thisCluster->pixel(k).y),(float)thisCluster->pixel(k).adc);
    //
    //
    // for (int ny = padSize; ny>0; --ny)
    // {
    //   for(int nx = 0; nx<padSize; nx++)
    //   {
    //     int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
    //     hitPads[j].push_back(hClust.GetBinContent(n));
    //   }
    // }


    // //Pad Initialization
    for (int iP = 0; iP < padSize*padSize*cnnLayers; ++iP)
      vPad[iP + doubOffset + j*padSize*padSize*cnnLayers] = 0.0;

    for (int k = 0; k < thisCluster->size(); ++k)
    {
      int thisX = int(-(float)thisCluster->pixel(k).x + xC + padHalfSize);
      int thisY = int(-(float)thisCluster->pixel(k).y + yC + padHalfSize);
      vPad[padOffset + thisX + thisY * padSize + doubOffset] = (float)thisCluster->pixel(k).adc;
    }


  }

  // for (int nx = 0; nx < padSize*padSize; ++nx)
  //     inHitPads[layerIds[0]][nx] = hitPads[0][nx];
  // for (int nx = 0; nx < padSize*padSize; ++nx)
  //     outHitPads[layerIds[1]][nx] = hitPads[1][nx];

  // std::cout << "Inner hit layer : " << inSeq << " - " << layerIds[0]<< std::endl;
  //
  // for(int i = 0; i < cnnLayers; ++i)
  // {
  //   std::cout << i << std::endl;
  //   auto thisOne = inHitPads[i];
  //   for (int nx = 0; nx < padSize; ++nx)
  //     for (int ny = 0; ny < padSize; ++ny)
  //     {
  //       std::cout << thisOne[ny + nx*padSize] << " ";
  //     }
  //     std::cout << std::endl;
  //
  // }
  //
  // std::cout << "Outer hit layer : " << outSeq << " - " << layerIds[1]<< std::endl;
  // for(int i = 0; i < cnnLayers; ++i)
  // {
  //   std::cout << i << std::endl;
  //   auto thisOne = outHitPads[i];
  //   for (int nx = 0; nx < padSize; ++nx)
  //     for (int ny = 0; ny < padSize; ++ny)
  //     {
  //       std::cout << thisOne[ny + nx*padSize ] << " ";
  //     }
  //     std::cout << std::endl;
  //
  // }
  //
  // std::cout << "TF Translation" << std::endl;
  // for(int i = 0; i < cnnLayers*2; ++i)
  // {
  //   std::cout << i << std::endl;
  //   int theOffset = i*padSize*padSize;
  //   for (int nx = 0; nx < padSize; ++nx)
  //     for (int ny = 0; ny < padSize; ++ny)
  //     {
  //       std::cout << vPad[(ny + nx*padSize) + theOffset + doubOffset] << " ";
  //     }
  //     std::cout << std::endl;
  //
  // }

  zZero = (siHits[0]->globalState()).position.z();
  zZero -= (float)(siHits[0]->globalPosition().perp()) * (deltaZ/deltaR); //copyDoublets.r(iD,layers[0])

  vLab[iLab + infoOffset] = deltaA   ; iLab++;
  vLab[iLab + infoOffset] = deltaADC ; iLab++;
  vLab[iLab + infoOffset] = deltaS   ; iLab++;
  vLab[iLab + infoOffset] = deltaR   ; iLab++;
  vLab[iLab + infoOffset] = deltaPhi ; iLab++;
  vLab[iLab + infoOffset] = deltaZ   ; iLab++;
  vLab[iLab + infoOffset] = zZero    ; iLab++;

  // std::cout << "iLab = "<<iLab << std::endl;

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session, { { "hit_shape_input", inputPads }, { "info_input", inputFeat } },
                { "output/Softmax" }, &outputs);

  return (outputs[0].flat<float>().data()[1]>t);
}


void HitPairGeneratorFromLayerPair::hitPairs(
					     const TrackingRegion & region, OrderedHitPairs & result,
					     const edm::Event& iEvent, const edm::EventSetup& iSetup, Layers layers) {

  auto const & ds = doublets(region, iEvent, iSetup, layers);
  for (std::size_t i=0; i!=ds.size(); ++i) {
    result.push_back( OrderedHitPair( ds.hit(i,HitDoublets::inner),ds.hit(i,HitDoublets::outer) ));
  }
  if (theMaxElement!=0 && result.size() >= theMaxElement){
     result.clear();
    edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
  }
}

HitDoublets HitPairGeneratorFromLayerPair::doublets( const TrackingRegion& region,
                                                     const edm::Event & iEvent, const edm::EventSetup& iSetup, const Layer& innerLayer, const Layer& outerLayer,
                                                     LayerCacheType& layerCache) {

  const RecHitsSortedInPhi & innerHitsMap = layerCache(innerLayer, region, iSetup);
  if (innerHitsMap.empty()) return HitDoublets(innerHitsMap,innerHitsMap);

  const RecHitsSortedInPhi& outerHitsMap = layerCache(outerLayer, region, iSetup);
  if (outerHitsMap.empty()) return HitDoublets(innerHitsMap,outerHitsMap);
  HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));
  doublets(region,
	   *innerLayer.detLayer(),*outerLayer.detLayer(),
	   innerHitsMap,outerHitsMap,iSetup,theMaxElement,result);

  return result;

}

void HitPairGeneratorFromLayerPair::doublets(const TrackingRegion& region,
						    const DetLayer & innerHitDetLayer,
						    const DetLayer & outerHitDetLayer,
						    const RecHitsSortedInPhi & innerHitsMap,
						    const RecHitsSortedInPhi & outerHitsMap,
						    const edm::EventSetup& iSetup,
						    const unsigned int theMaxElement,
						    HitDoublets & result){

  tensorflow::setLogging("0");
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/lustre/home/adrianodif/CNNDoublets/freeze_models/layer_map_model_final_nonorm.pb");
  tensorflow::Session* session = tensorflow::createSession(graphDef,16);
  std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31};

  int inSeq = innerHitDetLayer.seqNum(), outSeq = outerHitDetLayer.seqNum();
  int inLay  = (int)(find(pixelDets.begin(),pixelDets.end(),innerHitDetLayer.seqNum()) - pixelDets.begin());
  int outLay = (int)(find(pixelDets.begin(),pixelDets.end(),outerHitDetLayer.seqNum()) - pixelDets.begin());
  bool doInference = (inLay < 10 && inLay > -1 && outLay < 10 && outLay > -1);
  doInference = false;
  // std::cout << "doInference = "<< doInference << std::endl;
  float t_ = 0.5;

  //  HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));
  typedef RecHitsSortedInPhi::Hit Hit;
  InnerDeltaPhi deltaPhi(outerHitDetLayer, innerHitDetLayer, region, iSetup);

  // std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << outerLayer.detLayer()->seqNum() << std::endl;

  // constexpr float nSigmaRZ = std::sqrt(12.f);
  constexpr float nSigmaPhi = 3.f;
  for (int io = 0; io!=int(outerHitsMap.theHits.size()); ++io) {
    if (!deltaPhi.prefilter(outerHitsMap.x[io],outerHitsMap.y[io])) continue;
    Hit const & ohit =  outerHitsMap.theHits[io].hit();
    PixelRecoRange<float> phiRange = deltaPhi(outerHitsMap.x[io],
					      outerHitsMap.y[io],
					      outerHitsMap.z[io],
					      nSigmaPhi*outerHitsMap.drphi[io]
					      );

    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(&innerHitDetLayer, ohit, iSetup, &outerHitDetLayer,
						       outerHitsMap.rv(io),outerHitsMap.z[io],
						       outerHitsMap.isBarrel ? outerHitsMap.du[io] :  outerHitsMap.dv[io],
						       outerHitsMap.isBarrel ? outerHitsMap.dv[io] :  outerHitsMap.du[io]
						       );
    if(!checkRZ) continue;

    Kernels<HitZCheck,HitRCheck,HitEtaCheck> kernels;

    auto innerRange = innerHitsMap.doubleRange(phiRange.min(), phiRange.max());
    LogDebug("HitPairGeneratorFromLayerPair")<<
      "preparing for combination of: "<< innerRange[1]-innerRange[0]+innerRange[3]-innerRange[2]
				      <<" inner and: "<< outerHitsMap.theHits.size()<<" outter";
    for(int j=0; j<3; j+=2) {
      auto b = innerRange[j]; auto e=innerRange[j+1];
      bool ok[e-b];
      switch (checkRZ->algo()) {
	case (HitRZCompatibility::zAlgo) :
	  std::get<0>(kernels).set(checkRZ);
	  std::get<0>(kernels)(b,e,innerHitsMap, ok);
	  break;
	case (HitRZCompatibility::rAlgo) :
	  std::get<1>(kernels).set(checkRZ);
	  std::get<1>(kernels)(b,e,innerHitsMap, ok);
	  break;
	case (HitRZCompatibility::etaAlgo) :
	  std::get<2>(kernels).set(checkRZ);
	  std::get<2>(kernels)(b,e,innerHitsMap, ok);
	  break;
      }
      for (int i=0; i!=e-b; ++i) {
	if (!ok[i]) continue;
	if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
	  delete checkRZ;
	  return;
	}
        if(doInference)
        {
          bool doit = makeInference(innerHitsMap.theHits[b+i].hit(),outerHitsMap.theHits[io].hit(),session,inSeq,outSeq,inLay,outLay,t_);
          if(doit)
            result.add(b+i,io);
        }
        else
          result.add(b+i,io);
      }
    }
    delete checkRZ;
  }
  LogDebug("HitPairGeneratorFromLayerPair")<<" total number of pairs provided back: "<<result.size();
  result.shrink_to_fit();

}
