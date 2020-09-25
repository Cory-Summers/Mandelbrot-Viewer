#pragma once
#include "strategy-interface-class.hpp"
#include "application.hpp"
class NormalState :
    public IStateStrategy
{
public:
  NormalState(Application* parent) : IStateStrategy(parent) {}
  virtual std::size_t HandleEvent(sf::Event&, Mandelbrot::Renderer&);
  virtual std::size_t HandleKeyboard(sf::Event&, Mandelbrot::Renderer&);
  virtual std::size_t Update(Mandelbrot::Renderer&);
  virtual StateInfo   Type() const { return StateInfo::Normal; }
private:
  void SuperSample(Mandelbrot::Renderer&, std::uint8_t const);
  void Screenshot(Mandelbrot::Renderer&);
};

