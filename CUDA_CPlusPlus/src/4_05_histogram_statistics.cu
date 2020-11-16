#include <stdio.h>
#include <cuda_runtime.h>

// 考虑到 cudaMemcpy 传输事件，等于或者大于 CPU 计算的时间。
// 使用 共享内存 来避免数据拷贝传输的问题。

// 需要处理的元素数量
#define SIZE 1000
// 图像灰度等级划分为 16
#define NUM_BIN 16

// Define kernel function.
__global__ void hist_without_atomic(int *device_b, int *device_a)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int item = device_a[tid];
  if (tid < SIZE)
  {
    device_b[item]++;
  }
}

__global__ void hist_with_atomic(int *device_b, int *device_a)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int item = device_a[tid];
  if (tid < SIZE)
  {
    atomicAdd(&(device_b[item]), 1);
  }
}

int main(int argc, char **argv)
{
  int host_a[SIZE];
  for (int i = 0; i < SIZE; ++i)
  {
    host_a[i] = i % NUM_BIN;
  }

  int host_b[NUM_BIN];
  for (int j = 0; j < NUM_BIN; ++j)
  {
    host_b[j] = 0;
  }

  int *device_a, *device_b;

  cudaMalloc((void**)&device_a, SIZE * sizeof(int));
  cudaMalloc((void**)&device_b, NUM_BIN * sizeof(int));

  cudaMemcpy(device_a, host_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

  // hist_without_atomic <<< (SIZE + NUM_BIN - 1) / NUM_BIN, NUM_BIN >>> (device_b, device_a);
  hist_with_atomic <<< (SIZE + NUM_BIN - 1) / NUM_BIN, NUM_BIN >>> (device_b, device_a);

  cudaMemcpy(host_b, device_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Histogram using 16 bin without shared Memory is: \n");
  for (int i = 0; i < NUM_BIN; ++i) 
  {
		printf("bin %d: count %d\n", i, host_b[i]);
  }
  
  cudaFree(device_a);
  cudaFree(device_b);

  return 0;
}