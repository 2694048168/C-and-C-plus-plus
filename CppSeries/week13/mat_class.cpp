/**
 * @file mat_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief why Class Templates
 * @attention
 *
 */

#include <iostream>

// Class IntMat
class IntMat
{
private:
    size_t rows;
    size_t cols;
    int *data;

public:
    IntMat(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        data = new int[rows * cols]{};
    }
    ~IntMat()
    {
        delete[] data;
    }
    IntMat(const IntMat &) = delete;
    IntMat &operator=(const IntMat &) = delete;
    int getElement(size_t r, size_t c);
    bool setElement(size_t r, size_t c, int value);
};

int IntMat::getElement(size_t r, size_t c)
{
    if (r >= this->rows || c >= this->cols)
    {
        std::cerr << "Indices are out of range\n";
        return 0;
    }
    return data[this->cols * r + c];
}

bool IntMat::setElement(size_t r, size_t c, int value)
{
    if (r >= this->rows || c >= this->cols)
        return false;

    data[this->cols * r + c] = value;
    return true;
}

// Class FloatMat
class FloatMat
{
    size_t rows;
    size_t cols;
    float *data;

public:
    FloatMat(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        data = new float[rows * cols]{};
    }
    ~FloatMat()
    {
        delete[] data;
    }
    FloatMat(const FloatMat &) = delete;
    FloatMat &operator=(const FloatMat &) = delete;
    float getElement(size_t r, size_t c);
    bool setElement(size_t r, size_t c, float value);
};

float FloatMat::getElement(size_t r, size_t c)
{
    if (r >= this->rows || c >= this->cols)
    {
        std::cerr << "getElement(): Indices are out of range\n";
        return 0.f;
    }
    return data[this->cols * r + c];
}

bool FloatMat::setElement(size_t r, size_t c, float value)
{
    if (r >= this->rows || c >= this->cols)
    {
        std::cerr << "setElement(): Indices are out of range\n";
        return false;
    }
    data[this->cols * r + c] = value;
    return true;
}

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    IntMat imat(3, 4);
    imat.setElement(1, 2, 256);
    FloatMat fmat(2, 3);
    fmat.setElement(1, 2, 3.14159f);

    // FloatMat fmat2(fmat); //error

    // FloatMat fmat3(2,3);
    // fmat3 = fmat; //error

    std::cout << imat.getElement(1, 2) << std::endl;
    std::cout << fmat.getElement(1, 2) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ mat_class.cpp
 * $ clang++ mat_class.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */