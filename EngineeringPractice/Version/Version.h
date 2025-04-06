#ifndef VERSION_H
#define VERSION_H

#include <string_view>

namespace Version{
    constexpr std::string_view git_hash = "08395bbdd37f417f102c39a14f5e7effab7bf580";
    constexpr std::string_view git_tag = "08395bb";
    constexpr std::string_view git_branch = "master";
    constexpr std::string_view git_commit_time = "2025-03-29 23:22:24";
    constexpr std::string_view build_time = "2025-04-06 22:20:01";
};

#endif // VERSION_H
