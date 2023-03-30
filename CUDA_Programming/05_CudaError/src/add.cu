#include "add.cuh"

#include <iostream>

// kernel function
__global__ void add(const double *array_x, const double *array_y, 
                    double *array_z, const size_t arraySize)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < arraySize)
    {
        // array_z[idx] = array_x[idx] + array_y[idx];
        /* call 'device function' */
        element_wise(array_x[idx], array_y[idx], array_z[idx]);
    }
}

// host function
__host__ void check(const double *host_arrayZ, const int arraySize)
{
    bool isError = false;
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (fabs(host_arrayZ[i] - value_c) > EPSILON)
        {
            isError = true;
            break;
        }
    }

    if (isError)
    {
        std::cout << "[Error]: There is numerical precision error.\n";
    }
    else
    {
        std::cout << "Finishing array add by element-wise sucessfully.\n";
    }
}

// device function
__device__ void element_wise(const double x, const double y, double &z)
{
    z = x + y;
}
