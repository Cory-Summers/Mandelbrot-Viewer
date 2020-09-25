#include "kernel.cuh"
////////////////////////////////////////////////////////////
/// These functions come from 
/// https://stackoverflow.com/questions/18455414/how-to-do-power-of-complex-number-in-cublas
using rtype = double;
using ctype = cuDoubleComplex;
#define rpart(x)   (cuCreal(x))
#define ipart(x)   (cuCimag(x))
#define cmplx(x,y) (make_cuDoubleComplex(x,y))

__host__ __device__ rtype carg(const ctype& z) { return (rtype)atan2(ipart(z), rpart(z)); } // polar angle
__host__ __device__ rtype cabs(const ctype& z) { return (rtype)cuCabs(z); }
__host__ __device__ ctype cp2c(const rtype d, const rtype a) { return cmplx(d * cos(a), d * sin(a)); }
__host__ __device__ ctype cpow(const ctype& z, const int& n) { return cmplx((pow(cabs(z), n) * cos(n * carg(z))), (pow(cabs(z), n) * sin(n * carg(z)))); }
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Source: https://solarianprogrammer.com/2013/02/28/mandelbrot-set-cpp-11/
/// \brief Returns smoothed color based on the number of dwells to max dwells
///
/// \return struct containing the rgb values to placed in buffer
////////////////////////////////////////////////////////////
__device__ RGB GetSmoothedColor(int n);
__global__ void MandelbrotKernel(PixelBuffer p_buffer, MandelPlotArea* plot_area)
{
  const int x = threadIdx.x + blockDim.x * blockIdx.x; //Position within image area
  const int y = threadIdx.y + blockDim.y * blockIdx.y; //
  //Image buffer must be reversed as it starts at the bottom not the top for iteration

  int dwells; //number of dwells per pixel
  RGB color;
  if (x < plot_area->width && y < plot_area->height)
  {
    dwells = PixelDwell(x, y, plot_area);
    color = GetSmoothedColor(dwells);
    //Places pixel data in buffer
    //Messy might refactor later
    *(p_buffer + (x + y * plot_area->width) * 4 + 0) = color.r;
    *(p_buffer + (x + y * plot_area->width) * 4 + 1) = color.g;
    *(p_buffer + (x + y * plot_area->width) * 4 + 2) = color.b;
    *(p_buffer + (x + y * plot_area->width) * 4 + 3) = 0xff;
  }

}
int __device__ PixelDwell(int const x, int const y, MandelPlotArea* plot_area)
{
  ctype c = cmplx(x, y), 
        z;
  size_t iter = 0;
  c = cmplx(
    c.x / (double)plot_area->width * (plot_area->x_max - plot_area->x_min) + plot_area->x_min,
    c.y / (double)plot_area->height * (plot_area->y_bot - plot_area->y_top) + plot_area->y_top
  );
  z = cmplx(c.x, c.y);
  while (iter < K_MAX_DWELL && cuCabs(z) < 2.0)
  {
    z = cuCadd(cpow(z, 2), c);
    ++iter;
  }
  return iter;
}

void KernelCall(PixelBuffer buffer, MandelPlotArea* cu_plot, MandelPlotArea* h_plot)
{
  dim3 bs(64, 4), grid(DivUp(h_plot->width, bs.x), DivUp(h_plot->height, bs.y));
  MandelbrotKernel <<<grid, bs >>> (buffer, cu_plot);
}

__device__ RGB GetSmoothedColor(int n) {
  // map n on the 0..1 interval
  RGB rgb;
  double t = (double)n / (double)K_MAX_DWELL;

  // Use smooth polynomials for r, g, b
  rgb.r = (int)(9 * (1 - t) * t * t * t * 255);
  rgb.g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
  rgb.b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
  return rgb;
}
