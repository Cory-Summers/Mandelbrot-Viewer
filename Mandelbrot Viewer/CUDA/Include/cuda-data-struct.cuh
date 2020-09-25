#ifndef CUDA_DATA_STRUCT_CUH
#define CUDA_DATA_STRUCT_CUH
#include "plot-area-struct.hpp"
#include <cstdint>
////////////////////////////////////////////////////////////
/// \brief Contains device side data for Rendering Kernel
////////////////////////////////////////////////////////////
struct CudaData
{
  std::uint8_t*   cu_buffer;     ///< pixel buffer
  std::size_t     cu_buffer_size;///< size of pixel buffer
  MandelPlotArea* cu_mandel_area;///< Pointer plot data for renderer
  bool init;
};
////////////////////////////////////////////////////////////
/// \brief Dividing up for Kernel Call.
////////////////////////////////////////////////////////////
constexpr int DivUp(int const & a, int const & b)
{ 
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

////////////////////////////////////////////////////////////
/// \brief Allocates device side buffer and plot struct
///
/// \param data Struct containing members to be allocated
/// \param buf_size Desired buffer size in bytes
///
/// \return If succeeds will return 0
////////////////////////////////////////////////////////////
int CuInitializeData(CudaData* data, std::size_t const& buf_size);

////////////////////////////////////////////////////////////
/// \brief Updates device side plot from host
///
/// \param data Device side struct to be updated
/// \param plot Host side plot
///
/// \return If succeeds will return 0
////////////////////////////////////////////////////////////
int CuUpdateData(CudaData& data, MandelPlotArea const& plot);

////////////////////////////////////////////////////////////
/// \brief Called to change size of device side buffer
///
/// \param data Device side struct be updated
/// \param width New width
/// \param height New height
///
/// \return If succeeds will return 0
////////////////////////////////////////////////////////////
int CuResizeBuffer(CudaData& data, int const& width, int const& height);

#endif CUDA_DATA_STRUCT_CUH
