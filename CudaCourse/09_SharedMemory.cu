/**
 * @file 09_SharedMemory.cu
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

// dynamic shared-memory
extern __shared__ float s_array[];

__global__ void kernel_1(float *d_A, const int N) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;

  // static shared-memory
  __shared__ float s_array[32];

  if (n < N) {
    s_array[tid] = d_A[n];
  }

  // 访问共享内存必须需要同步机制, 线程块内同步
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < 32; ++i) {
      printf("[INFO] kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
    }
  }
  //   printf("==============================\n\n");
}

__global__ void kernel_2(float *d_A, const int N) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;

  if (n < N) {
    s_array[tid] = d_A[n];
  }
  __syncthreads();

  if (tid == 0) {
    for (int i = 0; i < 32; ++i) {
      printf("[INFO] kernel_2: %f, blockIdx: %d\n", s_array[i], bid);
    }
  }
  //   printf("==============================\n\n");
}

// ----------------------------------------
int main(int argc, const char **argv) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "[INFO] 运行GPU设备: " << deviceProps.name << std::endl;

  int nElems = 64;
  int nbytes = nElems * sizeof(float);

  float *h_A = nullptr;
  h_A = (float *)malloc(nbytes);
  for (int i = 0; i < nElems; ++i) {
    h_A[i] = float(i);
  }

  float *d_A = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, nbytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

  dim3 block(32);
  dim3 grid(2);
  kernel_1<<<grid, block>>>(d_A, nElems);
  printf("==============================\n\n"); // !hahaha

  // dynamic shared-memory call kernel
  kernel_2<<<grid, block, 32>>>(d_A, nElems);
  printf("==============================\n\n"); // !hahaha

  CUDA_CHECK(cudaFree(d_A));
  free(h_A);
  CUDA_CHECK(cudaDeviceReset());

  return 0;
}