/**
 * @file mat_example.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief simple demo about matrix class in OpenCV.
 * @attention
 *
 */

#include <iostream>

#include <opencv2/core.hpp>

/**
 * @brief main function and entry point
 */
int main(int argc, char const *argv[])
{
    std::string s("Hello ");
    s += "C";
    s.operator+=(" and CPP!");

    std::cout << s << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 * 
 * if you install the OpenCV library, and then as following:
 * $ clang++ mat_example.cpp
 * $ clang++ mat_example.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 * if you NOT install the OpenCV library, and then as following:
 * $ git clone https://github.com/Microsoft/vcpkg.git
 * $ .\vcpkg\bootstrap-vcpkg.bat
 * $ vcpkg install opencv4:x64-windows
 * $ cmake -B build -G Ninja -A x64 -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
 * 
 */