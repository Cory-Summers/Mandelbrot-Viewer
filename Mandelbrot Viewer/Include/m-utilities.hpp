#pragma once
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <string>

namespace Utility
{
  using string_cref = std::string const&;
  using time_point = std::chrono::system_clock::time_point;
  std::string GetDateString(
    string_cref format = std::string("%Y_%m_%d__%H-%M-%S"), 
    time_point const & time_data = std::chrono::system_clock::now()
  );
  std::size_t GetScreenshotFilename();
  std::size_t StringToUll(std::string const&);
}