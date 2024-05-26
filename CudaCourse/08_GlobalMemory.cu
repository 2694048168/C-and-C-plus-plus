/**
 * @file 08_GlobalMemory.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utility.cuh"
#include <iostream>

// CUDA 全局静态变量(global memory)
__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void kernel(void) {
  d_y[0] += d_x;
  d_y[1] += d_x;

  printf("[Kernel function] d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0],
         d_y[1]);
}

// --------------------------------
int main(int argc, const char **) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "[INFO] 运行GPU设备: " << deviceProps.name << std::endl;

  int h_y[2] = {10, 20};
  CUDA_CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

  dim3 block(1);
  dim3 grid(1);
  kernel<<<grid, block>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
  printf("[Host result] h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

  CUDA_CHECK(cudaDeviceReset());

  return 0;
}
