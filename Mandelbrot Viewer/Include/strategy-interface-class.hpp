#pragma once
#include <cstdint>
#include <iostream>
#include <SFML/Graphics.hpp>
#include "renderer-class.hpp"
class Application;
enum class StateInfo
{
  Nothing,
  Normal,
  ChangeState,
  Zooming
};
class IStateStrategy
{
public:
  virtual std::size_t HandleEvent(sf::Event &, Mandelbrot::Renderer &) = 0;
  virtual std::size_t HandleKeyboard(sf::Event &, Mandelbrot::Renderer &) = 0;
  virtual std::size_t Update(Mandelbrot::Renderer&) = 0;
protected:
  Application* m_parent;
  IStateStrategy(Application * parent = nullptr) : m_parent(parent) {}
};

