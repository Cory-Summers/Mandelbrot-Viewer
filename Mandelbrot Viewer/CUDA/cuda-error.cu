#include "cuda-error.cuh"
#include <iostream>
constexpr bool FAILURE = 1;
constexpr bool SUCCESS = 0;

bool _CudaErrorCheck(cudaError_t const& error)
{
  if (error != cudaSuccess)
  {
    std::cerr << cudaGetErrorName(error) << "> " << cudaGetErrorString(error) << '\n';
    return FAILURE;
  }
  return SUCCESS;
}