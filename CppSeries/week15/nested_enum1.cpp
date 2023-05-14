/**
 * @file nested_enum1.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Nested type and Nested Classes
 * @attention
 *
 */

#include <iostream>

enum DataType
{
    TYPE8U,
    TYPE8S,
    TYPE32F,
    TYPE64F
};
class Mat
{
private:
    DataType type;
    void *data;

public:
    Mat(DataType type) : type(type), data(NULL) {}

    DataType getType() const { return type; }
};

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    Mat image(TYPE8U);

    if (image.getType() == TYPE8U)
        std::cout << "This is an 8U matrix." << std::endl;
    else
        std::cout << "I am not an 8U matrix." << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ nested_enum1.cpp
 * $ clang++ nested_enum1.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */