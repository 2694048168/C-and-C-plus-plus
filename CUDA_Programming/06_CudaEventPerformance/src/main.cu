#include "cudaError.cuh"
#include "add.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

// -------------------------------------
int main(int argc, char const *argv[])
{
    const size_t arraySize = 1000000;
    const size_t bytesSize = sizeof(Precision) * arraySize;
    Precision *array_X = new Precision[arraySize]();
    Precision *array_Y = new Precision[arraySize]();
    Precision *array_Z = new Precision[arraySize]();

    for (size_t i = 0; i < arraySize; ++i)
    {
        array_X[i] = a;
        array_Y[i] = b;
    }

    /* CPU performacne
    ------------------------- */
    std::cout << "The performance of CPU\n";
    cpu_performance(array_X, array_Y, array_Z, arraySize);
    check(array_Z, arraySize);
    std::cout << "---------------------------------------\n";

    /* memory copy from HOST into DEVICE GPU.
    ------------------------------------------ */
    Precision *d_X, *d_Y, *d_Z;
    CHECK(cudaMalloc(&d_X, bytesSize));
    CHECK(cudaMalloc(&d_Y, bytesSize));
    CHECK(cudaMalloc(&d_Z, bytesSize));

    CHECK(cudaMemcpy(d_X, array_X, bytesSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Y, array_Y, bytesSize, cudaMemcpyHostToDevice));

    /* GPU performacne
    ------------------------- */
    std::cout << "The performance of GPU\n";
    gpu_performance(d_X, d_Y, d_Z, arraySize);
    CHECK(cudaMemcpy(array_Z, d_Z, bytesSize, cudaMemcpyDeviceToHost));
    check(array_Z, arraySize);
    std::cout << "---------------------------------------\n";

    delete [] array_X;
    delete [] array_Y;
    delete [] array_Z;

    CHECK(cudaFree(d_X));
    CHECK(cudaFree(d_Y));
    CHECK(cudaFree(d_Z));

    return 0;
}
