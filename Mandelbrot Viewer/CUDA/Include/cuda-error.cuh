#ifndef CUDA_ERROR_CUH
#define CUDA_ERROR_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

bool _CudaErrorCheck(cudaError_t const& error);

#define CudaErrorCheck(var, function) \
var = function; \
if(_CudaErrorCheck(error)) { return 1; }

#endif //CUDA_ERROR_CUH