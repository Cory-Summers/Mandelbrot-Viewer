#include "zoom-state-strat.hpp"

std::size_t ZoomState::HandleEvent(sf::Event& ev, Mandelbrot::Renderer& renderer)
{
  switch (ev.type)
  {
  case sf::Event::MouseMoved:
    RectangleUpdate(sf::Mouse::getPosition(m_parent->GetWindow()));
    break;
  case sf::Event::MouseButtonReleased:
    if (ev.mouseButton.button == sf::Mouse::Button::Left)
    {
      RectangleUpdate(sf::Mouse::getPosition(m_parent->GetWindow()));
      UpdatePlot(renderer);
      break;
    }
  }
  return std::size_t();
}

std::size_t ZoomState::HandleKeyboard(sf::Event&, Mandelbrot::Renderer&)
{
  return std::size_t();
}

std::size_t ZoomState::Update(Mandelbrot::Renderer& renderer)
{
  m_parent->GetWindow().draw(m_rectangle);
  return 0;
}

void ZoomState::Initialize()
{
  m_rectangle = sf::RectangleShape({ 1.f, 1.f });
  m_rectangle.setOutlineThickness(2.f);
  m_rectangle.setOutlineColor({ 0xFF, 0xFF, 0xFF, 0x7F });
  m_rectangle.setFillColor(sf::Color::Transparent);
  m_rectangle.setPosition(sf::Vector2f(m_origin));
  aspect_ratio = static_cast<float>(m_parent->GetWindow().getSize().y) / m_parent->GetWindow().getSize().x;
}

void ZoomState::RectangleUpdate(sf::Vector2i const& mouse_pos)
{
  const float delta(std::abs(static_cast<float>(mouse_pos.x - m_origin.x)) * 2.0f);
  const sf::Vector2f size(delta, delta * aspect_ratio);
  m_rectangle.setSize(size);
  m_rectangle.setPosition(sf::Vector2f(m_origin) - (size / 2.0f));
}

void ZoomState::UpdatePlot(Mandelbrot::Renderer& renderer)
{
  std::cout << m_rectangle.getSize().x << '\n';
  if (m_rectangle.getSize().x <= 30.f) { return; }
  MandelPlotArea new_plot = renderer.GetPlot();
  const auto plot_dim = new_plot.GetDimensions();
  const double plot_ratio  = m_rectangle.getSize().x / new_plot.width;
  Vec2   dist_scr =
  { m_rectangle.getPosition().x / new_plot.width,
    m_rectangle.getPosition().y / new_plot.height
  };
  new_plot.x_min += (dist_scr.x * plot_dim.x);
  new_plot.y_top -= (dist_scr.y * plot_dim.y);
  new_plot.y_bot  = (new_plot.y_top - (plot_ratio * plot_dim.y));
  new_plot.x_max  = (new_plot.x_min + (plot_ratio * plot_dim.x));
  renderer.UpdatePlot(new_plot);
  renderer.Render();
}
