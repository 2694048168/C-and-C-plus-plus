/**CUDA 并行执行具有分层结构
 * 即每次内核启动可以被切分为多个并行执行的块，同时每一个块可以被切分为多个并行执行的线程。
 * 0、启动 N 个 block，每一个 block 只有 1 个 thread。
 * 1、启动 1 个 block，每一个 block 中有 N 个 thread。
 * 2、启动 N/2 个 block，每个 block 中有 N/2 个 thread。
 * 
 * The number of blocks and threads have been limitted for each hardware GPU.
 * 设备的 maxThreadPerBlock 属性限制每个 block 能启动的最大线程数量 RTX 2060 = 1024，常设 512、1024.
 * 设备的最大能启动的块数量被限制为 2^31 - 1 = 2,147,483,647.
 * gpu_add << <((N + 511) / 512), 512>> > (device_a, device_b, device_c);
 * gpu_add << <((N + 1023) / 1024), 1024>> > (device_a, device_b, device_c);
 * 技巧Tips：除法向上取整操作。使用取模操作一样能够达到相同的效果。
 *
 * 注意 x 维度限制计算是以上的情况，而对于 y 和 z 维度方向的计算限制 block_number = 65535.
 */

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// Defining the number of elements in array.
#define N 100000000

/**Compute the unique ID of thread.
 * Formula: unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
 * 解析：
 *   threadIdx.x 命令得到该线程在该块中的 ID 值；
 *   blockIdx.x 命令得到任意线程在当前块 ID 值；
 *   blockDim.x 命令得到该块中的线程总数；
 * 由于 block 和 thread 就组合就类似与一个二维矩阵形式，称之为 grid 
 * 这样 blockIdx.x * blockDim.x 结果相当于一个偏移量，再加上 threadIdx.x 结果就是在二维矩阵中的唯一索引总 ID。
 *   
 * tid 每次更新在 x 维度方向，blockDim.x * gridDim.x 结果就类似计算二维矩阵的行数和列数相乘，即已经启动的线程总数。
 * blockDim.x 表示 x 维度方向上块中的线程数量；gridDim.x 表示 x 维度方向上启动的块的数量。
 * tid 每次更新都加上这样一个偏移量值，则就是下一个任务/进程的索引。
 */
// Define the kernel function.
__global__ void gpu_add(int *device_a, int *device_b, int *device_c)
{
  // Getting the index of current kernel.
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N)
  {
    device_c[tid] = device_b[tid] + device_a[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char **argv)
{
  // Declare host and decive arrays.
  int *host_a, *host_b, *host_c;
  int *device_a, *device_b, *device_c;

  // Allocate memory on host and device.
  host_a = (int*)malloc(N * sizeof(int));
  host_b = (int*)malloc(N * sizeof(int));
  host_c = (int*)malloc(N * sizeof(int));

  cudaMalloc((void**)&device_a, N * sizeof(int));
  cudaMalloc((void**)&device_b, N * sizeof(int));
  cudaMalloc((void**)&device_c, N * sizeof(int));

  // Initialize host arrays.
  for (unsigned int i = 0; i < N; ++i)
  {
    host_a[i] = 2 * i * i;
    host_b[i] = i;
  }

  // Copy data from host memory to device memory.
  cudaMemcpy(device_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Kernel call
  clock_t start_device_gpu = clock();
  // gpu_add << <((N + 511) / 512), 512>> > (device_a, device_b, device_c);
  gpu_add << <((N + 1023) / 1024), 1024>> > (device_a, device_b, device_c);

  // Copy data from device memory to host memory.
  cudaMemcpy(host_c, device_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  // This ensures that kernel execution is finishes before going forward.
  // << UNIX 环境高级编程 >>第三版书籍中有介绍 sync 这种操作。
  cudaDeviceSynchronize();

  clock_t end_device_gpu = clock();
  // Cost compute time on GPU.
  double time_device_gpu = (double)(end_device_gpu - start_device_gpu) / CLOCKS_PER_SEC;

  // 测试计算结果是否正确。
  int correct = 1;
  printf("Vector addition on GPU.\n");
  for (unsigned int i = 0; i < N; ++i)
  {
    if ((host_a[i] + host_b[i] != host_c[i]))
    {
      correct = 0;
    }
  }

  if (correct == 1)
  {
    printf("GPU has computed vector sum correctly.\n");
  }
  else
  {
    printf("There is an error in GPU computation.\n");
  }

  // 当然这样方式计算时间并不能完全体现 GPU 性能，
  // 因为 GPU 启动需要预热时间，一般都有一个预热测试程序来启动GPU。
  std::cout << N << " of elements in array." << std::endl;
  std::cout << "Device GPU time: " << time_device_gpu << std::endl;

  // Free up the host memory and device memory.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  free(host_a);
  free(host_b);
  free(host_c);

  return 0;
}
