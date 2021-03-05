#include "tfAOT.h"
#include <iostream>
#include <cmath>
#include <ctime>

int main()
{
  std::srand(static_cast<unsigned int>(time(0)));
  const int input_hit_size = 1 * 16 * 16 * 24;
  const int input_info_size = 1 * 67;
  const int output_size = 1 * 2;
  float input_hit[input_hit_size] = {0};
  float input_info[input_info_size] = {0};
  float output[2];
  float MAX=static_cast<float>(RAND_MAX);
  for(int i=0;i<input_hit_size;i++)
    input_hit[i]=std::rand()/MAX;
  for(int i=0;i<input_info_size;i++)
    input_info[i]=std::rand()/MAX;
  tfAOT::model Model;
  Model.run(input_hit,input_info,output);
  std::cout<<output[0]<<','<<output[1]<<std::endl;
  return 0;
}
