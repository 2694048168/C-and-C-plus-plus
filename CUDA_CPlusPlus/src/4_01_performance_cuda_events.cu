/**CDUA 中高级编程概念
 * 1、测量 CUDA 程序的性能：CDUA events、NVIDIA Visual Profiler.
 * 2、CUDA 中错误处理：从代码中进行处理、CUDA-GDB 调试器/NSight.
 * 3、CUDA 程序性能的提升：使用适当的块和线程数量、最大化数学运算效率、使用合并的或跨步式的访存、避免 warp 内分支、使用锁定页面的内存.
 * 4、CUDA 流：使用多个 CUDA 流.
 */

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/**CPU时间度量性能取决于高精度的定时器；GPU kernel 是异步运行的
 * CUDA Events 是在 CUDA 应用运行的特定时刻被记录的时间戳。
 *
 * C/C++ 的 API 参数分为入参和出参，
 * 入参就是函数所需要的使用的参数；出参就是函数所需要返回的参数。
 * CUDA API 返回值都是用于标志该操作的成功或者错误；而将需要的返回参数作为参数列表传入。
 * C++ 不支持返回多个返回值，故采用参数列表作为返回；而 Python 支持返回多个返回值。
 */

// 数学运算效率 = 数学运算操作 / 访存操作
// 提升程序性能，前提分析程序的瓶颈在哪里。

// Define the constant variables.
#define N 50000000  // The number of elements in array.

// Define kernel function.
__global__ void gpuAdd(int *device_a, int *device_b, int *device_c)
{
  // Getting the thread index of current kernel.
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N)
  {
    device_c[tid] = device_a[tid] + device_b[tid];
    // 偏移量
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char **argv)
{
  // Defining host arrays using Dynamic Memory Allocation.
  int *host_a, *host_b, *host_c;
  host_a = (int*)malloc(N * sizeof(int));
  host_b = (int*)malloc(N * sizeof(int));
  host_c = (int*)malloc(N * sizeof(int));
  
  // Define device pointers.
  int *device_a, *device_b, *device_c;

  // CUDA Events.
  // 定义 CUDA 事件类型变量。
  cudaEvent_t event_start, event_stop;
  // 创建 CUDA 事件。
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  // 记录 CUDA 事件。
  cudaEventRecord(event_start, 0);

  // Allocate thr memory on device.
  cudaMalloc((void**)&device_a, N * sizeof(int));
  cudaMalloc((void**)&device_b, N * sizeof(int));
  cudaMalloc((void**)&device_c, N * sizeof(int));

  // Initialize arrays.
  for (int i = 0; i < N; ++i)
  {
    host_a[i] = 2 * i * i;
    host_b[i] = i;
  }

  // Copy input data from host to device memory.
  cudaMemcpy(device_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Call kernel passing device pointers as parameters.
  gpuAdd <<< 512, 512 >>> (device_a, device_b, device_c);

  // Copy result back to host memory from device memory.
  cudaMemcpy(host_c, device_c, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // 记录 CUDA 事件。
  cudaEventRecord(event_stop, 0);
  cudaEventSynchronize(event_stop);
  // 定义变量用于计算 CUDA 事件，度量性能。
  float event_lapsed_time;
  cudaEventElapsedTime(&event_lapsed_time, event_start, event_stop);
  printf("Time to add %d numbers: %3.lf ms.\n", N, event_lapsed_time);

  // 验证 GPU 计算结果。
  int correct_flag = 1;
  std::cout << "Vector addition on GPU.\n";
  for (int i = 0; i < N; ++i)
  {
    if (host_a[i] + host_b[i] != host_c[i])
    {
      correct_flag = 0;
    }
  }
  if (correct_flag == 1)
  {
    std::cout << "GPU has computed sum correctly.\n";
  }
  else
  {
    std::cout << "There is an error in GPU computation.\n";
  }

  // Free up host Dynamic Memory.
  free(host_a);
  free(host_b);
  free(host_c);
  // Free up memory on device.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}