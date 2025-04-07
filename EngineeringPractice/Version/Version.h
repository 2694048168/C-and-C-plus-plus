#ifndef VERSION_H
#define VERSION_H

#include <string_view>

namespace Version{
    constexpr std::string_view git_hash = "4f63c8123e9697afbd848a543e95aae86b2061bb";
    constexpr std::string_view git_tag = "4f63c81";
    constexpr std::string_view git_branch = "master";
    constexpr std::string_view git_commit_time = "2025-04-06 22:22:00";
    constexpr std::string_view build_time = "2025-04-07 21:58:05";
};

#endif // VERSION_H
