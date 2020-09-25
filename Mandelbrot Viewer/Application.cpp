#include "application.hpp"
#include "normal-state-strat.hpp"
#include "m-utilities.hpp"
#include "zoom-state-strat.hpp"
Application::Application(int, char * argv[])
  : m_window()
  , m_event()
  , m_background()
  , m_renderer()
  , m_state(State::Normal)
  , m_strategy(std::make_unique<NormalState>(this))
{
  Initialize();
  m_background.setOutlineColor(sf::Color::Red);
  m_background.setOutlineThickness(5.0f);
}

void Application::TriggerStateChange()
{
  switch (m_state)
  {
  case State::Normal:
    m_strategy = std::make_unique<NormalState>(this);
    break;
  case State::ZoomRect:
    m_strategy = std::make_unique<ZoomState>(this, sf::Mouse::getPosition(m_window));
    break;
  }
}

void Application::UpdateBackground()
{
  m_background.setTexture(m_renderer.GetTexture().get());
}

int Application::Run()
{
  while (m_window.isOpen())
  {
    HandleEvent();
    m_window.clear();
    m_window.draw(m_background);
    m_strategy->Update(m_renderer);
    m_window.display();
  }
  return 0;
}


void Application::Resize(sf::Vector2f const& size)
{
  m_renderer.Resize(sf::Vector2u(size));
  m_background.setSize(size);
  m_background.setTextureRect({ 0,0, static_cast<int>(size.x), static_cast<int>(size.y) });
  m_window.setView(sf::View({ 0.f, 0.f, size.x, size.y }));
}

void Application::Initialize()
{
  m_window.create(sf::VideoMode(3000, 2000), "Mandelbrot Viewer");
  m_background.setSize({ 3000, 2000 });
  m_window.setFramerateLimit(30);
  m_renderer.Initialize(m_window.getSize());
  m_background.setTexture(m_renderer.GetTexture().get());
}

void Application::HandleEvent()
{
  while (m_window.pollEvent(m_event))
  {
    switch(m_event.type){
      case sf::Event::Closed:
        m_window.close();
        break;
      case sf::Event::KeyPressed:
        HandleKeyPress();
        break;
      case sf::Event::KeyReleased:
        HandleKeyRelease();
        break;
      case sf::Event::MouseButtonPressed:
        if (m_renderer.Ready()) {
          std::cout << m_renderer.GetState() << std::endl;
          m_strategy = std::make_unique<ZoomState>(this, sf::Mouse::getPosition(m_window));
        }
        break;
      case sf::Event::MouseButtonReleased:
        m_strategy->HandleEvent(m_event, m_renderer);
        m_strategy = std::make_unique<NormalState>(this);
        break;
      case sf::Event::Resized:
        Resize(sf::Vector2f(m_event.size.width, m_event.size.height));
        break;
      default:
        m_strategy->HandleEvent(m_event, m_renderer);
        break;
    }
  }
}

void Application::HandleKeyPress()
{
  switch (m_event.key.code)
  {
  case sf::Keyboard::C:
    if (m_event.key.control)
      m_window.close();
    break;
  default:
    m_strategy->HandleKeyboard(m_event, m_renderer);
  }
}

