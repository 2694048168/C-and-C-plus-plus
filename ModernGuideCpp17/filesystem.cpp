/**
 * @file filesystem.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstdlib> /* for EXIT_FAILURE */
#include <filesystem>
#include <iostream>

/**
 * @brief filesystem in C++17
 * @attention 根据参数依赖查找(argument dependent lookup, ADL), 
 * 不需要使用完全限定的名称来调用 is_regular_file()、file_size()、is_directory()、
 * exists()等函数; 它们都属于命名空间 std::filesystem, 但是因为它们的参数也属于这个命名空间,
 * 所以调用它们时会自动在这个命名空间中进行查找.
 *
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <filepath> \n";
        return EXIT_FAILURE;
    }

    std::filesystem::path fp{argv[1]};
    if (is_regular_file(fp))
    {
        std::cout << fp << " exists with " << file_size(fp) << " bytes\n";
    }
    else if (is_directory(fp))
    {
        std::cout << fp << " is a directory containing:\n";
        for (const auto &e : std::filesystem::directory_iterator{fp})
        {
            std::cout << " " << e.path() << "\n";
        }
    }
    else if (exists(fp))
    {
        std::cout << fp << " is a special file\n";
    }
    else
    {
        std::cout << "path " << fp << " does not exist\n";
    }

    return 0;
}