#include "normal-state-strat.hpp"

std::size_t NormalState::HandleEvent(sf::Event& event, Mandelbrot::Renderer& renderer)
{
  std::size_t ret = 0;
  return ret;
}

std::size_t NormalState::HandleKeyboard(sf::Event& ev, Mandelbrot::Renderer& renderer)
{
  std::size_t ret = 0;

  switch (ev.key.code)
  {
  case sf::Keyboard::P:
    if (ev.key.control)
      renderer.SuperSample(4);
    else
      Screenshot(renderer);
    break;
  case sf::Keyboard::E:
  case sf::Keyboard::Q:
  case sf::Keyboard::Dash:
  case sf::Keyboard::Equal:
    if (renderer.Ready())
      renderer.Zoom((ev.key.code == sf::Keyboard::Q) || (ev.key.code == sf::Keyboard::Dash));
    break;
  case sf::Keyboard::W:
  case sf::Keyboard::A:
  case sf::Keyboard::S:
  case sf::Keyboard::Left:
  case sf::Keyboard::Right:
  case sf::Keyboard::Up:
  case sf::Keyboard::Down:
    if (renderer.Ready())
      renderer.Move(ev.key.code);
    break;
  }
  return std::size_t();
}

std::size_t NormalState::Update(Mandelbrot::Renderer& renderer)
{
  if (renderer.GetState() == (int)Mandelbrot::Renderer::State::Done)
  {
    renderer.SwapBuffer();
    m_parent->UpdateBackground();
  }
}

void NormalState::Screenshot(Mandelbrot::Renderer & renderer)
{
  std::cout << "Saving Screenshot...\n";
  std::async(
    std::launch::async,
    Mandelbrot::Renderer::SaveImage,
    renderer.GetTexture()->copyToImage()
  );
}