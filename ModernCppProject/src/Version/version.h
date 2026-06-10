#pragma once

#include <string_view>

// clang-format off
namespace Ithaca::version {
constexpr inline int VERSION_MAJOR = 1;
constexpr inline int VERSION_MINOR = 0;
constexpr inline int VERSION_PATCH = 0;

constexpr std::string_view VERSION_STRING = "1.0.0";
constexpr std::string_view BUILD_TIMESTAMP = "2026-06-10 21:16:00";
constexpr std::string_view VERSION_BUILD_STRING = "1.0.0-2026-06-10 21:16:00";
} // namespace Ithaca::version

// clang-format on
