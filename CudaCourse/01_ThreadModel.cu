/**
 * @file 01_ThreadModel.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdio>

__global__ void hello_cuda_kernel() {
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int unique_identifier_id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello CUDA from block %d and thread %d, global id %d\n", block_id,
         thread_id, unique_identifier_id);
}

// -----------------------------------
int main(int argc, const char **argv) {
  hello_cuda_kernel<<<2, 4>>>();

  cudaDeviceSynchronize();

  return 0;
}
