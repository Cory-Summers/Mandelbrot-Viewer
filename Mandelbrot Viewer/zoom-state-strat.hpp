#pragma once
#include "strategy-interface-class.hpp"
#include "application.hpp"
class ZoomState : public IStateStrategy
{
public:
  ZoomState(Application* parent, sf::Vector2i const& start) : IStateStrategy(parent) { m_origin = start; Initialize(); }
  virtual std::size_t HandleEvent(sf::Event&, Mandelbrot::Renderer&);
  virtual std::size_t HandleKeyboard(sf::Event&, Mandelbrot::Renderer&);
  virtual std::size_t Update(Mandelbrot::Renderer&);
  virtual StateInfo   Type() const { return StateInfo::Zooming; }
private:
  void Initialize();
  void RectangleUpdate(sf::Vector2i const &);
  void UpdatePlot(Mandelbrot::Renderer &);
  sf::RectangleShape m_rectangle;
  sf::Vector2i m_origin;
  float aspect_ratio;
};

