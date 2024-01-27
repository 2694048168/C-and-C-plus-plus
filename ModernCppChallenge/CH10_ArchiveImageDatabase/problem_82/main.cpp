/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Creating a PNG that represents a national flag
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pngwriter.h"

#include <iostream>
#include <string>
#include <string_view>

/**
 * @brief Creating a PNG that represents a national flag
 * 
 * Write a program that generates a PNG file that represents the national flag of Romania,
 * shown here. The size of the image in pixels, as well as the path to
 *  the destination file, should be provided by the user:
 * 
 */

/**
 * @brief Solution: pngwriter library in C++
 https://github.com/pngwriter/pngwriter
------------------------------------------------------ */
void create_flag(const int width, const int height, std::string_view filepath)
{
    pngwriter flag{width, height, 0, filepath.data()};

    const int size = width / 3;
    // red rectangle
    flag.filledsquare(0, 0, size, 2 * size, 65535, 0, 0);
    // yellow rectangle
    flag.filledsquare(size, 0, 2 * size, 2 * size, 65535, 65535, 0);
    // blue rectangle
    flag.filledsquare(2 * size, 0, 3 * size, 2 * size, 0, 0, 65535);

    flag.close();
}

// ------------------------------
int main(int argc, char **argv)
{
    int         width = 0, height = 0;
    std::string filepath;

    std::cout << "Width: ";
    std::cin >> width;

    std::cout << "Heigh: ";
    std::cin >> height;

    std::cout << "Output: ";
    std::cin >> filepath;

    create_flag(width, height, filepath);

    return 0;
}
