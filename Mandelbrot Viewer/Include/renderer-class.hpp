#pragma once
#include <atomic>
#include <deque>
#include <future>
#include <memory>
#include <thread>
#include <SFML/Graphics.hpp>

#include "kernel.cuh"
#include "cuda-data-struct.cuh"
#include <cuda-error.cuh>
namespace Mandelbrot {
  class Renderer
  {
  public:
    using texture_ptr = std::shared_ptr<sf::Texture>;
    using future_ptr = std::future<texture_ptr>;
    enum class State
    {
      Ready,
      Running,
      Done
    };
    Renderer();
    void Initialize(sf::Vector2u const& window_size);
    void Render();
    void Resize(sf::Vector2u const &);
    void Move(sf::Keyboard::Key const&, double rate = .25);
    void UpdatePlot(MandelPlotArea const&);
    void SuperSample(std::uint8_t const& factor);
    int  GetState() const { return render_state.load(); }
    void SetState(State const& st) { render_state.store((int)st); }
    bool SwapBuffer();
    bool Ready() const;
    void Zoom(bool in_out, double rate = .75);
    texture_ptr GetTexture() { return m_texture; }
    std::size_t HandleEvent(sf::Event const& ev) { return 0; }
    static void SaveImage(sf::Image);
    MandelPlotArea const& GetPlot() const { return m_plot; }
  private:
    void SuperSampleCall(std::uint8_t const, MandelPlotArea const);
    int CreateCudaBuffers();
    texture_ptr CudaCall();
    MandelPlotArea m_plot;
    CudaData m_data;
    std::uint8_t*     host_buffer;
    std::atomic<int> render_state;
    future_ptr m_future;
    texture_ptr m_texture;
    std::deque<MandelPlotArea> m_history;

  };
};

