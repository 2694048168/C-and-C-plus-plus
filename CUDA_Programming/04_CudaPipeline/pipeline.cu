#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

const double EPSILON = 1.0e-15;
const double value_a = 1.23;
const double value_b = 2.34;
const double value_c = 3.57; /* c = a + b */

// kernel function
__global__ void add(const double *array_x, const double *array_y, 
                    double *array_z, const size_t arraySize);
// host function
__host__ bool check(const double *host_arrayZ, const int arraySize);

/* 核函数可以调用不带执行配置的自定义函数,
  这样的自定义函数称为设备函数 device function,
  它是在设备中执行, 并在设备中被调用的;
  与之相比, 核函数是在设备中执行, 但在主机端被调用的.
  ------------------------------------------- */
__device__ void element_wise(const double x, const double y, double &z);


int main(int argc, char const *argv[])
{
    // Z = X + Y by element-wise 
    const size_t arraySize = 1000000;
    const size_t sizeBytes = sizeof(double) * arraySize;
    double *host_arrayX = (double*)malloc(sizeBytes);
    double *host_arrayY = (double*)malloc(sizeBytes);
    double *host_arrayZ = (double*)malloc(sizeBytes);
    // TODO: the malloc successful?
    for (size_t idx = 0; idx < arraySize; ++idx)
    {
        host_arrayX[idx] = value_a;
        host_arrayY[idx] = value_b;
    }
    
    double *device_arrayX, *device_arrayY, *device_arrayZ;
    cudaMalloc(&device_arrayX, sizeBytes);
    cudaMalloc(&device_arrayY, sizeBytes);
    cudaMalloc(&device_arrayZ, sizeBytes);
    // TODO: the malloc successful?
    cudaMemcpy(device_arrayX, host_arrayX, sizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_arrayY, host_arrayY, sizeBytes, cudaMemcpyHostToDevice);

    const size_t block_size = 64;
    // const size_t grid_size = arraySize / block_size;
    const size_t grid_size = (arraySize % block_size == 0)
                            ? (arraySize / block_size)
                            : (arraySize / block_size + 1);
    add<<<grid_size, block_size>>>(device_arrayX, device_arrayY, 
                                    device_arrayZ, arraySize);

    cudaMemcpy(host_arrayZ, device_arrayZ, sizeBytes, cudaMemcpyDeviceToHost);
    bool isError = check(host_arrayZ, arraySize);
    if (isError)
    {
        std::cout << "[Error]: There is numerical precision error.\n";
    }
    else
    {
        std::cout << "Finishing array add by element-wise sucessfully.\n";
    }
    
    // free memeoy
    free(host_arrayX);
    free(host_arrayY);
    free(host_arrayZ);
    cudaFree(device_arrayX);
    cudaFree(device_arrayY);
    cudaFree(device_arrayZ);
    
    return 0;
}

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
__host__ bool check(const double *host_arrayZ, const int arraySize)
{
    bool isError = false;
    for (size_t i = 0; i < arraySize; ++i)
    {
        /* 在判断两个浮点数是否相等时, 不能用运算符 ==
        而要将这两个数的差的绝对值与一个很小的数进行比较
        ----------------------------------------- */
        if (fabs(host_arrayZ[i] - value_c) > EPSILON)
        {
            isError = true;
            break;
        }
    }
    return isError;
}

// device function
__device__ void element_wise(const double x, const double y, double &z)
{
    z = x + y;
}
