#ifndef MANDELBROT_PLOT_HPP
#define MANDELBROT_PLOT_HPP
////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <array>
#include <cstdint>
#include <cmath>
////////////////////////////////////////////////////////////
/// \brief Easier access 2d vector on both host and device side.
///        Would use sf::vector2f but I don't believe it will work correctly
///        device side.
////////////////////////////////////////////////////////////
struct Vec2 : public std::array<double, 2>
{
private:
	using std::array<double, 2>::_Elems;
public:
	Vec2() : std::array<double, 2>() {}
	Vec2(double const& a, double const& b) : std::array<double, 2>({ a,b }) {}
	double& x = _Elems[0];
	double& y = _Elems[1];
};
////////////////////////////////////////////////////////////
/// \brief Area of the MandelProt to render along with W/H of Image
////////////////////////////////////////////////////////////
struct MandelPlotArea
{
	int width;  ///< Pixel Width of Plot
	int height; ///< Pixel Height of Plot
	double x_min; ///< Left Side of Plot Area
	double x_max; ///< Right Side of Plot Area
	double y_top; ///< Top of Plot Area
	double y_bot; ///< Bottom of Plot Area
	////////////////////////////////////////////////////////////
	/// \brief Get the center of the Plot Area
	///
	/// \return (x, y) point of center
	////////////////////////////////////////////////////////////
	Vec2 GetCenter() const noexcept;

	////////////////////////////////////////////////////////////
	/// \brief Get Size of the Plot Area
	///
	/// \return (x,y) size of Plot Area
	////////////////////////////////////////////////////////////
	Vec2 GetDimensions() const noexcept;
	////////////////////////////////////////////////////////////
	/// \brief Returns the byte size of the buffer for display area
  ////////////////////////////////////////////////////////////
	std::size_t BufferSize() const noexcept { return static_cast<std::size_t>(height) * width * 4ull; }
};
inline Vec2 MandelPlotArea::GetCenter() const noexcept
{
	return { (x_max + x_min) / 2.0, (y_bot + y_top) / 2.0 };
}
inline Vec2 MandelPlotArea::GetDimensions() const noexcept
{
	return { std::abs(x_max - x_min), std::abs(y_bot - y_top) };
}
#endif //MANDELBROT_PLOT_HPP