#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining number of elements in array.
#define N 100

/**kernel function
 * 每一个块中的这个线程，用当前块的ID来初始化 tid 变量。
 * 根据 tid 每一个线程将一对元素相加，
 * 如果块的总数等于每一个数组中元素的总数，那么所有加法操作将并行完成。
 * 每一个线程通过 blockIdx.x 内置变量来知道自己的 ID，并使用这个 ID 来索引数组，计算每一对元素相加。
 */
// Defining kernel function for vector addition.
__global__ void gpu_add(int *device_a, int *device_b, int *device_c)
{
  // Getting block index of current kernel.
  int tid = blockIdx.x; // handle the data at this index.
  if (tid < N)
  {
    
    device_c[tid] = device_a[tid] + device_a[tid];
  }
}

int main(int argc, char **argv)
{
  // Defining host arrays.
  int host_a[N], host_b[N], host_c[N];

  // Defining device pointers.
  int *device_a, *device_b, *device_c;

  // Allocate the memory.
  cudaMalloc((void**)&device_a, N * sizeof(int));
  cudaMalloc((void**)&device_b, N * sizeof(int));
  cudaMalloc((void**)&device_c, N * sizeof(int));

  // Initializing arrays.
  for (unsigned int i = 0; i < N; ++i)
  {
    host_a[i] = 2 * i;
    host_b[i] = i;
  }

  // Copy input arrays from host to device memory.
  cudaMemcpy(device_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Calling kernels with N blocks and one thread per block, passing device pointers as paramerters.
  gpu_add << <N, 1>> > (device_a, device_b, device_c);

  // Copy result back to host memory form device memory.
  cudaMemcpy(host_c, device_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Vector result on GPU\n");
  for (unsigned int i = 0; i < N; ++i)
  {
    printf("The sum of %d element is %d + %d = %d\n", i, host_a[i], host_b[i], host_c[i]);
  }

  // Free up memory.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}