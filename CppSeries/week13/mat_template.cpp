/**
 * @file mat_template.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Class Templates
 * @attention
 *
 */

#include <iostream>

// Class Template
template <typename T>
class Mat
{
private:
    size_t rows;
    size_t cols;
    T *data;

public:
    Mat(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        data = new T[rows * cols]{};
    }
    ~Mat()
    {
        delete[] data;
    }

    Mat(const Mat &) = delete;
    Mat &operator=(const Mat &) = delete;
    T getElement(size_t r, size_t c);
    bool setElement(size_t r, size_t c, T value);
};

template <typename T>
T Mat<T>::getElement(size_t r, size_t c)
{
    if (r >= this->rows || c >= this->cols)
    {
        std::cerr << "getElement(): Indices are out of range\n";
        return 0;
    }
    return data[this->cols * r + c];
}

template <typename T>
bool Mat<T>::setElement(size_t r, size_t c, T value)
{
    if (r >= this->rows || c >= this->cols)
    {
        std::cerr << "setElement(): Indices are out of range\n";
        return false;
    }

    data[this->cols * r + c] = value;
    return true;
}

template class Mat<int>; // Explicitly instantiate template Mat<int>
// template Mat<float> and Mat<double> will be instantiate implicitly

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    Mat<int> imat(3, 4);
    imat.setElement(1, 2, 256);
    Mat<float> fmat(2, 3);
    fmat.setElement(1, 2, 3.14159f);
    Mat<double> dmat(2, 3);
    dmat.setElement(1, 2, 2.718281828);

    // Mat<float> fmat2(fmat); //error

    // Mat<float> fmat3(2,3);
    // fmat3 = fmat; //error

    std::cout << imat.getElement(1, 2) << std::endl;
    std::cout << fmat.getElement(1, 2) << std::endl;
    std::cout << dmat.getElement(1, 2) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ mat_template.cpp
 * $ clang++ mat_template.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */