#pragma once
#include <atomic>
#include <SFML/Graphics.hpp>
#include <future>
#include "renderer-class.hpp"
#include "strategy-interface-class.hpp"
class Application
{
  ////////////////////////////////////////////////////////////
  /// \brief Internal arguments parsing currently only deals with
  ///  sizing of window currently
  ////////////////////////////////////////////////////////////
  struct Arguments
  {
    sf::Vector2u width;
  };
public:
  enum class State {
    Normal,
    ZoomRect
  };
  Application(int, char * argv[]);
  sf::RenderWindow& GetWindow() { return m_window; }
  void TriggerStateChange();
  void UpdateBackground();
  int Run();
  void TakeScreenshot();
private:
  void Resize(sf::Vector2f const&);
  void Initialize();

  void HandleEvent();
  void HandleKeyPress();
  void HandleKeyRelease() {}

  void ChangeState() {}
  Mandelbrot::Renderer m_renderer;
  sf::RectangleShape m_background;
  sf::RenderWindow m_window;
  std::unique_ptr<IStateStrategy> m_strategy;
  sf::Event m_event;
  State m_state;
};

