#ifndef MANDELBROT_KERNEL_CUH
#define MANDELBROT_KERNEL_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cstdint>

#include "plot-area-struct.hpp"
#include "cuda-data-struct.cuh"
#include "cuda-error.cuh"
#define K_MAX_DWELL 512 //Leaving as define as to be able modify while compiling
using PixelBuffer = std::uint8_t *;
struct RGB { std::uint8_t r, g, b; };
////////////////////////////////////////////////////////////
/// \brief Allows for easy calling from non cuda files for easier linking.
///       Without it there were linking errors.
///
/// \param buffer Device pixel buffer for returning image
/// \param cu_plot Device side plot info
/// \param h_plot  Host side plot info
////////////////////////////////////////////////////////////
void KernelCall(PixelBuffer buffer, MandelPlotArea* cu_plot, MandelPlotArea* h_plot);
////////////////////////////////////////////////////////////
/// \brief Main kernel call to render Mandelbrot Image in p_buffer
///
/// \param p_buffer Pointer to device pixel buffer where the rgba values of each pixel
///        is stored.
/// \param plot_area Specifies the area of the Mandelbrot set to be rendered into p_buffer
////////////////////////////////////////////////////////////
__global__ void MandelbrotKernel(PixelBuffer p_buffer, MandelPlotArea* plot_area);
////////////////////////////////////////////////////////////
/// \brief Calculates the number of dwells on a single pixel
///
/// \param x Pixel position on the x dimension
/// \param y Pixel position on the y dimension
/// \param plot_area Pointer to data determining area to render and size of the image
///
/// \return Number of dwells done
////////////////////////////////////////////////////////////
__device__ int PixelDwell(int const x, int const y, MandelPlotArea* plot_area);
#endif //MANDELBROT_KERNEL_CUH

