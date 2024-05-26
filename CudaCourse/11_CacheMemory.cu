/**
 * @file 11_CacheMemory.cu
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

__global__ void kernel(void) { printf("The kernel function call\n"); }

// --------------------------------
int main(int argc, const char **) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "[INFO] 运行GPU设备: " << deviceProps.name << std::endl;

  if (deviceProps.globalL1CacheSupported) {
    std::cout << "[INFO] 支持全局内存L1缓存" << std::endl;
  } else {
    std::cout << "[INFO] 不支持全局内存L1缓存" << std::endl;
  }
  std::cout << "[INFO] L2缓存大小: " << deviceProps.l2CacheSize / (1024 * 1024)
            << "M" << std::endl;

  dim3 block(1);
  dim3 grid(1);
  kernel<<<grid, block>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
