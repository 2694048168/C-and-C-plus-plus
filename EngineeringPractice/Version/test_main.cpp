#include "Version.h"

#include <format>
#include <iostream>

// g++ test_main.cpp -std=c++20
// clang++ test_main.cpp -std=c++20
// --------------------------------------
int main(int argc, const char *argv[])
{
    std::cout << "CMake and Git Version Track" << std::endl;
    std::cout << std::format("Git Hash ---> {}\n", Version::git_hash);
    std::cout << std::format("Git Tag ---> {}\n", Version::git_tag);
    std::cout << std::format("Git Branch ---> {}\n", Version::git_branch);
    std::cout << std::format("Git Commit DateTime ---> {}\n", Version::git_commit_time);
    std::cout << std::format("Git Build DateTime ---> {}\n", Version::build_time);

    return 0;
}
