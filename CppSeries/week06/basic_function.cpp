/**
 * @file basic_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the basic function in C++
 * @attention parameters and arguments
 *
 */

#include <iostream>
#include <cfloat>

struct Matrix
{
    int rows;
    int cols;
    float *pData;
};

float matrix_max(struct Matrix mat)
{
    float max = FLT_MIN;
    // find max value of mat
    for (int r = 0; r < mat.rows; r++)
        for (int c = 0; c < mat.cols; c++)
        {
            float val = mat.pData[r * mat.cols + c];
            max = (max > val ? max : val);
        }
    return max;
}

Matrix *create_matrix(int rows, int cols)
{
    Matrix *p = new Matrix{rows, cols};
    p->pData = new float[p->rows * p->cols]{1.f, 2.f, 3.f};
    // you should check if the memory is allocated successfully
    return p;
}

bool matrix_add(const Matrix &matA, const Matrix &matB, Matrix &matC)
{
    // check the dimensions of the three matrices
    // re-create matC if needed
    // do: matC = matA + matB
    // return true if everything is right
    return true;
}

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. dynamic memory with malloc and free
    ----------------------------------------------- */
    Matrix matA = {3, 4};
    matA.pData = new float[matA.rows * matA.cols]{1.f, 2.f, 3.f};

    Matrix matB = {4, 8};
    matB.pData = new float[matB.rows * matB.cols]{10.f, 20.f, 30.f};

    Matrix matC = {4, 2};
    matC.pData = new float[matC.rows * matC.cols]{100.f, 200.f, 300.f};

    // some operations on the matrices

    float maxa = matrix_max(matA);
    float maxb = matrix_max(matB);
    float maxc = matrix_max(matC);

    std::cout << "max(matA) = " << maxa << std::endl;
    std::cout << "max(matB) = " << maxb << std::endl;
    std::cout << "max(matC) = " << maxc << std::endl;

    delete[] matA.pData;
    delete[] matB.pData;
    delete[] matC.pData;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ basic_function.cpp
 * $ clang++ basic_function.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */