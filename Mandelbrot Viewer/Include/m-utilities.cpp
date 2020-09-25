#include "m-utilities.hpp"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <regex>
namespace fs = std::filesystem;
namespace Utility
{
  std::size_t GetLastFileNumber();

}
std::string Utility::GetDateString(string_cref format, time_point const& time_data)
{
  std::stringstream ss;
  std::time_t c_time = std::chrono::system_clock::to_time_t(time_data);
  ss << std::put_time(std::localtime(&c_time), format.c_str());
  return ss.str();
}

std::size_t Utility::GetScreenshotFilename()
{
  static std::atomic<std::size_t> value = GetLastFileNumber();
  return value++;
}

std::size_t Utility::StringToUll(std::string const& str)
{
  std::size_t ret;
  std::stringstream ss;
  ss << str;
  ss >> ret;
  return ret;
}
std::size_t Utility::GetLastFileNumber()
{
  static const std::regex file_rx("(screenshot)([0-9]+)(\\.png)");
  fs::path directory(fs::current_path().string() + "/screenshots/");
  std::smatch match;
  std::string str;
  std::vector<std::size_t> matches;
  for (auto& entry : fs::directory_iterator(directory))
  {
    str = entry.path().filename().string();
    if (std::regex_match(str, match, file_rx))
      matches.push_back(StringToUll(match[2]));
  }
  auto iter = std::max_element(matches.begin(), matches.end());
  return (iter == matches.end()) ? static_cast<std::size_t>(1) : (*iter + 1);
}
