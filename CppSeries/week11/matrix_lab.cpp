/**
 * @file unique_ptr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief smart pointer in C++11 
 * @attention 
 *
 */

#include <iostream>
#include <memory>
#include <string>

class Matrix
{
private:
    size_t rows;
    size_t cols;
    std::shared_ptr<float[]> data;

public:
    Matrix(size_t r, size_t c)
    {
        if (r * c == 0)
        {
            rows = 0;
            cols = 0;
            data = nullptr;
        }
        else
        {
            rows = r;
            cols = c;
            data = std::shared_ptr<float[]>(new float[r * c]);
        }
    }
    Matrix(const Matrix &m) : rows(m.rows), cols(m.cols), data(m.data) {}

    friend std::ostream &operator<<(std::ostream &os, const Matrix &m)
    {
        os << "size (" << m.rows << "x" << m.cols << ")" << std::endl;
        os << "[" << std::endl;
        for (size_t r = 0; r < m.rows; r++)
        {
            for (size_t c = 0; c < m.cols; c++)
                os << m.data[r * m.rows + c] << ", ";
            os << std::endl;
        }
        os << "]";
        return os;
    }
};

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    Matrix m1(3, 8);
    Matrix m2(4, 8);

    m2 = m1;

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ matrix_lab.cpp
 * $ clang++ matrix_lab.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */