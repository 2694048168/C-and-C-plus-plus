#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>

/**When N = 10000000, and then show nothing, may be the stack overflow.
 * suppose using new to Dynamic Memory Allocation, or using std::vector.
 * but what should I do?
 * 当 N 愈来愈大，那么 GPU 和 CPU 之间的处理时间相差就愈来愈大，这时候 GPU 并行处理的高效更能体现出来。
 * 当然，这个问题也提醒自己要时刻注意 堆栈的存储大小限制，毕竟控制内存等存储资源是必备的技能，
 * 而且出现这样堆栈溢出，编译器是很难检查出来的，编译通过，但是运行达不到预期效果。
 */
// Defining number of elements in array.
// #define N 10000
#define N 1000000000


// Defining vector addition function for single core CPU.
void cpu_add(int *host_a, int *host_b, int *host_c)
{
  int tid = 0;
  while (tid < N)
  {
    host_c[tid] = host_a[tid] + host_b[tid];
    tid ++;
  }
}

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
  // int tid = threadIdx.x; // handle the data at this index.
  if (tid < N)
  {
    device_c[tid] = device_a[tid] + device_a[tid];
  }
}

int main(int argc, char **argv)
{
  // Defining host arrays. N = 10000
  // int host_a[N], host_b[N], host_c[N];

  // Defining host arrays using Dynamic Memory Allocation. N = 1000000000
  int *host_a, *host_b, *host_c;
  host_a = (int*)malloc(N * sizeof(int));
  host_b = (int*)malloc(N * sizeof(int));
  host_c = (int*)malloc(N * sizeof(int));

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
  clock_t start_device_gpu = clock();
  // std::cout << "Doing on GPU for vector addition." << std::endl;
  gpu_add << <N, 1>> > (device_a, device_b, device_c);
  cudaThreadSynchronize();
  clock_t end_device_gpu = clock();
  // Cost compute time on GPU.
  double time_device_gpu = (double)(end_device_gpu - start_device_gpu) / CLOCKS_PER_SEC;

  // Calling CPU function for vector addition.
  clock_t start_host_cpu = clock();
  // std::cout << "Doing on CPU for vector addition." << std::endl;
  // cpu_add(host_a, host_b, host_c);
  cpu_add(host_a, host_b, host_c);

  clock_t end_host_gpu = clock();
  // Cost compute time on CPU.
  double time_host_cpu = (double)(end_host_gpu - start_host_cpu) / CLOCKS_PER_SEC;

  std::cout << N << " of elements in array." << std::endl;
  std::cout << "Host CPU time: " << time_host_cpu << std::endl;
  std::cout << "Device GPU time: " << time_device_gpu << std::endl;

  // Free up memory.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  free(host_a);
  free(host_b);
  free(host_c);

  return 0;
}