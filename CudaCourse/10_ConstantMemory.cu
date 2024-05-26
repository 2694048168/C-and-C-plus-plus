/**
 * @file 10_ConstantMemory.cu
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

// constant memory on GPU device
__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1(void) {
  //   __constant__ float c_data2 = 6.6f;
  printf("Constant data c_data = %.2f.\n", c_data);
}

// 传递数组大小 N, 位于device的 const-memory 中
// 读取一次, 广播到所有线程, 高效快速
__global__ void kernel_2(int N) {
  int idx = threadIdx.x;
  if (idx < N) {
  }
}

// -------------------------------
int main(int argc, const char **) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "[INFO] 运行GPU设备: " << deviceProps.name << std::endl;

  float h_data = 8.8f;
  CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

  dim3 block(1);
  dim3 grid(1);
  kernel_1<<<grid, block>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
  printf("Constant data h_data = %.2f.\n", h_data);

  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
