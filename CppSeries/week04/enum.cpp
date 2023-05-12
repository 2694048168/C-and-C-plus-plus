/**
 * @file enum.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief enum data type in C++.
 * @attention tricks
 *
 */

#include <iostream>
#include <cmath>
#include <cstdint>

enum color
{
    WHITE = 0,
    BLACK,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    NUM_COLORS /* trick for length of color enum. */
};

enum datatype
{
    TYPE_INT8 = 1,
    TYPE_INT16 = 2,
    TYPE_INT32 = 4,
    TYPE_INT64 = 8
};

struct Point
{
    enum datatype type;
    union
    {
        int8_t data8[3];
        int16_t data16[3];
        int32_t data32[3];
        int64_t data64[3];
    };
};

size_t datawidth(struct Point pt)
{
    return size_t(pt.type) * 3;
}

int64_t l1norm(struct Point pt)
{
    int64_t result = 0;
    switch (pt.type)
    {
    case (TYPE_INT8):
        result = std::abs(pt.data8[0]) +
                 std::abs(pt.data8[1]) +
                 std::abs(pt.data8[2]);
        break;
    case (TYPE_INT16):
        result = std::abs(pt.data16[0]) +
                 std::abs(pt.data16[1]) +
                 std::abs(pt.data16[2]);
        break;
    case (TYPE_INT32):
        result = std::abs(pt.data32[0]) +
                 std::abs(pt.data32[1]) +
                 std::abs(pt.data32[2]);
        break;
    case (TYPE_INT64):
        result = std::abs(pt.data64[0]) +
                 std::abs(pt.data64[1]) +
                 std::abs(pt.data64[2]);
        break;
    }
    return result;
}

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    enum color pen_color = RED;
    pen_color = color(3); /* convert 'int' to 'enum' */
    std::cout << "we have " << NUM_COLORS << " pens." << std::endl;

    // pen_color += 1; /* error */
    int color_index = pen_color;
    color_index += 1;
    std::cout << "color_index = " << color_index << std::endl;

    // declaration and initialization
    struct Point point1 = {.type = TYPE_INT8, .data8 = {-2, 3, 4}};
    struct Point point2 = {.type = TYPE_INT32, .data32 = {1, -2, 3}};

    std::cout << "Data width = " << datawidth(point1) << std::endl;
    std::cout << "Data width = " << datawidth(point2) << std::endl;

    std::cout << "L1 norm = " << l1norm(point1) << std::endl;
    std::cout << "L1 norm = " << l1norm(point2) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ enum.cpp
 * $ clang++ enum.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */