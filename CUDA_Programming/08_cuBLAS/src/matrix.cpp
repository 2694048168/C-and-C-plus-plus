#include "matrix.hpp"

#include <iostream>

void print_matrix(int row, int col, double *A, const char* name)
{
    std::cout << name << "\n";
    for (size_t i = 0; i < row; ++i)
    {
        for (size_t j = 0; j < col; ++j)
        {
            std::cout << A[j*row+i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}