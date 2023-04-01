#include "cudaError.cuh"
#include "matrix.hpp"

#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

// -----------------------------------
int main(int argc, char const *argv[])
{
    int M = 2, K = 3, N = 2;
    int MK = M * K;
    int KN = K * N;
    int MN = M * N;

    double *h_A = (double*)malloc(sizeof(double) * MK);
    double *h_B = (double*)malloc(sizeof(double) * KN);
    double *h_C = (double*)malloc(sizeof(double) * MN);
    for (size_t i = 0; i < MK; ++i)
    {
        h_A[i] = i;
    }
    print_matrix(M, K, h_A, "Matrix A");

    for (size_t i = 0; i < KN; ++i)
    {
        h_B[i] = i;
    }
    print_matrix(K, N, h_B, "Matrix B");

    for (size_t i = 0; i < MN; ++i)
    {
        h_C[i] = i;
    }

    /* the device GPU memory and copy
    ---------------------------------- */
    double *g_A, *g_B, *g_C;
    CHECK(cudaMalloc(&g_A, sizeof(double) * MK));
    CHECK(cudaMalloc(&g_B, sizeof(double) * KN));
    CHECK(cudaMalloc(&g_C, sizeof(double) * MN));

    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);
    cublasSetVector(MN, sizeof(double), h_B, 1, g_B, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, g_A, M, g_B, K, &beta, g_C, M);
    cublasDestroy(handle);

    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
    print_matrix(M, N, h_C, "Matrix C = A x B");

    // TODO: free
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaFree(g_A));
    CHECK(cudaFree(g_B));
    CHECK(cudaFree(g_C));
    
    return 0;
}
