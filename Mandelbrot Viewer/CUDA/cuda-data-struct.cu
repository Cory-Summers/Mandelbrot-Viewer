#include "cuda-data-struct.cuh"
#include "cuda-error.cuh"

int CuInitializeData(CudaData* data, std::size_t const& buffer_size)
{
  cudaError_t error;
  CudaErrorCheck(error, cudaMalloc(&(data->cu_buffer), buffer_size));
  CudaErrorCheck(error, cudaMalloc(&(data->cu_mandel_area), sizeof(MandelPlotArea)));
  data->init = true;
  data->cu_buffer_size = buffer_size;
  return 0;
}

int CuUpdateData(CudaData& data, MandelPlotArea const& plot)
{
  cudaError_t error;
  CudaErrorCheck(
    error, 
    cudaMemcpy(data.cu_mandel_area, &plot, sizeof(MandelPlotArea), cudaMemcpyHostToDevice)
  );
  return 0;
}

int CuResizeBuffer(CudaData& data, int const& width, int const& height)
{
  cudaError_t error;
  const std::size_t plot_size = (width * 4ull) * height;
  CudaErrorCheck(error, cudaFree(data.cu_buffer));
  CudaErrorCheck(error, cudaMalloc(&(data.cu_buffer), plot_size));
  data.cu_buffer_size = plot_size;
  return 0;
}
