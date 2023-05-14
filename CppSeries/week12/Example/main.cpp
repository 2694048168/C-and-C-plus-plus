/**
 * @file virtual.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief virtual function and pure-virtual function
 * @attention virtual table and polymorphic
 *
 */

#include <cstdio>

#include "mymatrix.hpp"

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    Matrix *matA = createMat(2, 3);
    Matrix *matB = createMat(2, 3);
    Matrix *matC = createMat(2, 3);
    Matrix *matD = createMat(3, 2);
    Matrix *matNULL = NULL;

    // initialization
    // You should have your own method to do it
    matA->data[3] = 2.3f;
    matB->data[3] = 3.1f;

    if (!add(matA, matB, matC))
        fprintf(stderr, "Matrix addition failed.");
    else
    {
        // You can have a better method to show the results
        printf("result=%f\n", matC->data[3]);
    }

    // more tests
    add(matA, matB, matD);

    add(matNULL, matB, matC);

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ main.cpp mymatrix.cpp
 * $ clang++ main.cpp mymatrix.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */