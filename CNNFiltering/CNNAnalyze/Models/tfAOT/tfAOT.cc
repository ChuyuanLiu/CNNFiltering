#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tfAOT.h"

namespace tfAOT
{
  model::model(int threads)
  {
    Eigen::ThreadPool *tp = new Eigen::ThreadPool(threads); 
    Eigen::ThreadPoolDevice *device = new Eigen::ThreadPoolDevice(tp, tp->NumThreads());
    Graph *graph = new Graph();
    graph->set_thread_pool(device);
    GraphPtr = graph;
  }
  void model::run(float *input_hit, float *input_info, float *output) 
  {
    const int output_size = 1 * 2;
    Graph *graph = static_cast<Graph*>(GraphPtr);
    graph->set_arg0_data(input_hit);
    graph->set_arg1_data(input_info);
    graph->Run();
    std::copy(graph->result0_data(), graph->result0_data() + output_size, output);
  }
}
