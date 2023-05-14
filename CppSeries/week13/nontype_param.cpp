/**
 * @file nontype_param.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Class Templates with non type parameters
 * @attention what non-type parameters
 *
 */

#include <iostream>

// Class Template
template <typename T, size_t rows, size_t cols>
class Mat
{
private:
    T data[rows][cols];

public:
    Mat() {}
    /* the default copy constructor will copy each element of
    a static array member, so we do not 'delete' the copy constructor,
    the same with the assignment operator */
    // Mat(const Mat&) = delete;
    // Mat& operator=(const Mat&) = delete;

    T getElement(size_t r, size_t c);
    bool setElement(size_t r, size_t c, T value);
};

template <typename T, size_t rows, size_t cols>
T Mat<T, rows, cols>::getElement(size_t r, size_t c)
{
    if (r >= rows || c >= cols)
    {
        std::cerr << "getElement(): indices are out of range\n";
        return 0;
    }
    return data[r][c];
}

template <typename T, size_t rows, size_t cols>
bool Mat<T, rows, cols>::setElement(size_t r, size_t c, T value)
{
    if (r >= rows || c >= cols)
    {
        std::cerr << "setElement(): Indices are out of range\n";
        return false;
    }

    data[r][c] = value;
    return true;
}

// Explicitly instantiate template Mat<int, 2, 2>
template class Mat<int, 2, 2>;
typedef Mat<int, 2, 2> Mat22i;
// template Mat<float, 3, 1> will be instantiate implicitly

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    Mat22i mat;

    mat.setElement(2, 3, 256);
    std::cout << mat.getElement(2, 3) << std::endl;

    mat.setElement(1, 1, 256);
    std::cout << mat.getElement(1, 1) << std::endl;

    Mat<float, 3, 1> vec;
    vec.setElement(2, 0, 3.14159f);
    std::cout << vec.getElement(2, 0) << std::endl;

    Mat<float, 3, 1> vec2(vec);
    std::cout << vec2.getElement(2, 0) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ nontype_param.cpp
 * $ clang++ nontype_param.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */