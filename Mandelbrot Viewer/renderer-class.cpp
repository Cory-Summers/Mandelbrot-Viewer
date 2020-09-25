#include "renderer-class.hpp"
#include <chrono>
#include <iostream>
#include "cuda-data-struct.cuh"
#include "kernel.cuh"
#include "cuda-error.cuh"
#include "m-utilities.hpp"
namespace Mandelbrot {
  constexpr std::array<double, 4> k_max_dim = { -2.0, 1.0, 1.0, -1.0 };
  Renderer::Renderer()
    : m_plot()
    , m_data()
    , render_state((int)State::Ready)
    , m_future()
    , m_texture(std::make_shared<sf::Texture>())
    , host_buffer(nullptr)
  {
  }
  void Renderer::Initialize(sf::Vector2u const& window_size)
  {
    m_plot =
    {
      static_cast<int>(window_size.x),
      static_cast<int>(window_size.y),
      k_max_dim[0],
      k_max_dim[1],
      k_max_dim[2],
      k_max_dim[3]
    };
    host_buffer = new std::uint8_t[m_plot.BufferSize()];
    CreateCudaBuffers();
    m_texture = CudaCall();
    render_state.store((int)State::Ready);
  }
  void Renderer::Render()
  {
    if (render_state.load() != (int)State::Running)
    {
      render_state.store((int)State::Running);
      m_future = std::async(std::launch::async, &Renderer::CudaCall, this);
    }
  }
  void Renderer::Resize(sf::Vector2u const& win_size)
  {
    if (render_state.load() == (int)State::Running)
      m_future.wait();
    const double aspect_ratio = static_cast<double>(win_size.y) / win_size.x;
    const double y_center = m_plot.GetCenter()[1];

    MandelPlotArea plot_area = m_plot;
    plot_area.width  = win_size.x;
    plot_area.height = win_size.y;
    plot_area.y_top  = y_center + m_plot.GetDimensions()[0] * aspect_ratio / 2.0;
    plot_area.y_bot =  y_center - m_plot.GetDimensions()[0] * aspect_ratio / 2.0;

    m_plot = plot_area;
    delete[] host_buffer;
    host_buffer = new std::uint8_t[m_plot.BufferSize()];
    CuResizeBuffer(m_data, win_size.x, win_size.y);

    Render();
  }
  void Renderer::Move(sf::Keyboard::Key const & key, double rate)
  {
    using Key = sf::Keyboard::Key;
    //Branchless for potental performance benefit
    const double move_amount = m_plot.GetDimensions()[0] * rate;
    const double x = move_amount * (-1.0 * (key == Key::Left || key == Key::A) + (key == Key::D || key == Key::Right));
    const double y = move_amount * (-1.0 * (key == Key::Down || key == Key::S) + (key == Key::W || key == Key::Up));
    m_plot.x_min += x;
    m_plot.x_max += x;
    m_plot.y_top += y;
    m_plot.y_bot += y;
    Render();
  }
  void Renderer::UpdatePlot(MandelPlotArea const& plot)
  {
    m_plot = plot;
  }
  void Renderer::SuperSample(std::uint8_t const& factor)
  {
    if (Ready()) {
      render_state.store((int)State::Running);
      std::async(std::launch::async, &Renderer::SuperSampleCall, this, factor, m_plot);
    }
  }
  bool Renderer::SwapBuffer() 
  {
    render_state.store((int)State::Ready);
    m_texture = m_future.get();
    return true;
  }
  bool Renderer::Ready() const
  {
    return ((State)GetState() != State::Running);
  }
  void Renderer::Zoom(bool in_or_out, double scale)
  {
    const auto center = m_plot.GetCenter();
    const auto dim = m_plot.GetDimensions();
    scale += 1.0 * in_or_out;
    //width times new scale then divide as adding to both sides
    const double w = (dim[0] * scale) / 2.0;
    const double y = (dim[1] * scale) / 2.0;
    m_plot.x_min = center[0] - w;
    m_plot.x_max = center[0] + w;
    m_plot.y_bot = center[1] - y;
    m_plot.y_top = center[1] + y;
    Render();
  }
  void Renderer::SaveImage(sf::Image image)
  { 
    //Supersample code needed
    const std::string file_path = "./screenshots/screenshot";
    const std::string extension = ".png";
    size_t i = Utility::GetScreenshotFilename();
    std::string file_name = file_path + std::to_string(i) + extension;
    image.saveToFile(file_name);

  }
  void Renderer::SuperSampleCall(std::uint8_t const sampling, MandelPlotArea const area)
  {
    sf::Image final_image, sample_image;
    std::uint8_t* pixel_buffer = new std::uint8_t[area.BufferSize()];
    MandelPlotArea sub_plot = area;
    double delta_x = area.GetDimensions().x / sampling;
    double delta_y = area.GetDimensions().y / sampling;
    final_image.create(area.width * sampling, area.height * sampling);
    for (std::uint8_t j = 0; j < sampling; ++j)
    {
      sub_plot.x_min = area.x_min;
      sub_plot.x_max = area.x_min + delta_x;
      sub_plot.y_top = area.y_top - (delta_y * j);
      sub_plot.y_bot = (area.y_top - delta_y) - (delta_y * j);
      for (std::uint8_t i = 0; i < sampling; ++i)
      {
        CuUpdateData(m_data, sub_plot);
        KernelCall(m_data.cu_buffer, m_data.cu_mandel_area, &sub_plot);
        cudaMemcpy(pixel_buffer, m_data.cu_buffer, m_data.cu_buffer_size, cudaMemcpyDeviceToHost);
        sample_image.create(area.width, area.height, pixel_buffer);
        final_image.copy(sample_image, static_cast<std::uint32_t>(area.width * i), static_cast<std::uint32_t>(area.height * j));
        sub_plot.x_min += delta_x;
        sub_plot.x_max += delta_x;
        std::cout << "Passing...\n";
      }
    }
    delete[] pixel_buffer;
    SaveImage(final_image);
    render_state.store((int)State::Ready);
  }
  int Renderer::CreateCudaBuffers()
  {
    int error = 0;
    error = CuInitializeData(&m_data, m_plot.BufferSize());
    if (error) return 1;
    return 0;
  }
  Renderer::texture_ptr Renderer::CudaCall()
  {
    auto start = std::chrono::steady_clock::now();
    texture_ptr texture = std::make_shared<sf::Texture>();
    cudaError_t error; 
    texture->create(m_plot.width, m_plot.height);
    CuUpdateData(m_data, m_plot);
    KernelCall(m_data.cu_buffer, m_data.cu_mandel_area, &m_plot);
    error = cudaMemcpy(host_buffer, m_data.cu_buffer, m_data.cu_buffer_size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Render Time - " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    texture->update(host_buffer);
    render_state.store((int)State::Done);
    return texture;
  }
};