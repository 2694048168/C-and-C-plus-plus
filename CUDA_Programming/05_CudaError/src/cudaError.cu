#include "cudaError.cuh"
#include "add.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

int main(int argc, char const *argv[])
{
    // Z = X + Y by element-wise 
    const size_t arraySize = 1000;
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
    CHECK(cudaMalloc(&device_arrayX, sizeBytes));
    CHECK(cudaMalloc(&device_arrayY, sizeBytes));
    CHECK(cudaMalloc(&device_arrayZ, sizeBytes));
    CHECK(cudaMemcpy(device_arrayX, host_arrayX, 
                    sizeBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_arrayY, host_arrayY, 
                    sizeBytes, cudaMemcpyHostToDevice));

    // const size_t block_size = 1280;
    const size_t block_size = 128;
    // const size_t grid_size = arraySize / block_size;
    const size_t grid_size = (arraySize % block_size == 0)
                            ? (arraySize / block_size)
                            : (arraySize / block_size + 1);
    add<<<grid_size, block_size>>>(device_arrayX, device_arrayY, 
                                    device_arrayZ, arraySize);
    /* 不能捕捉调用核函数的相关错误, 
    因为核函数不返回任何值(void 修饰)
    使用如下方式, 可以 Debug 到 kernel function Error.
    ----------------------------------------------- */
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize()); 
    /* export CUDA_LAUNCH_BLOCKING=1, kernel function 同步执行;
    CUDA 提供了名为 CUDA-MEMCHECK 的工具集,
    包括 memcheck、 racecheck、initcheck、 synccheck 共 4 个工具,
    用于检查内存错误:
    $ cuda-memcheck --tool memcheck [options] app_name [options]
    $ cuda-memcheck --tool racecheck [options] app_name [options]
    $ cuda-memcheck --tool initcheck [options] app_name [options]
    $ cuda-memcheck --tool synccheck [options] app_name [options]
    ------------------------------------------------------------- */

    // CHECK(cudaMemcpy(host_arrayZ, device_arrayZ,
    //                 sizeBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(host_arrayZ, device_arrayZ,
                    sizeBytes, cudaMemcpyDeviceToHost));
    check(host_arrayZ, arraySize);
    
    // free memeoy
    free(host_arrayX);
    free(host_arrayY);
    free(host_arrayZ);
    CHECK(cudaFree(device_arrayX));
    CHECK(cudaFree(device_arrayY));
    CHECK(cudaFree(device_arrayZ));
    
    return 0;
}
