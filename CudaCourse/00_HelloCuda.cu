/**
 * @file 00_HelloCuda.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-20
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdio>
// #include <iostream>

__global__ void hello_cuda_kernel() {
  /* 核函数(kernel function)中不支持 C++ 的 iostream . */
  // std::cout << "Hello CUDA from GPU." << std::endl;
  printf("Hello CUDA from GPU.\n");
}

// -------------------------------------
int main(int argc, char const *argv[]) {

  hello_cuda_kernel<<<2, 2>>>();

  /* CUDA 的运行时 API 函数 cudaDeviceSynchronize 的作用
  是同步主机(host)与设备(device), 所以能够促使缓冲区刷新,
  否则打印不出字符串. */
  cudaDeviceSynchronize();

  return 0;
}
