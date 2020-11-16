#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Define number of elements in array.
#define N 500000

// Define kernel function for vector addition.
__global__ void gpuAdd(int *device_a, int *device_b, int *device_c)
{
  // Getting thread index of current kernel.
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N)
  {
    device_c[tid] = device_a[tid] + device_b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main(int argc, char **argv)
{
  // Define host pointers.
  int *host_a, *host_b, *host_c;

  // Define device pointers for CUDA Stream 0.
  int *device_a_stream0, *device_b_stream0, *device_c_stream0;
  // Define device pointers for CUDA Stream 1.
  int *device_a_stream1, *device_b_stream1, *device_c_stream1;

  // CUDA Stream.
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // CUDA Event.
  cudaEvent_t event_start, event_stop;
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_stop);
  cudaEventRecord(event_start, 0);

  // malloc function 分配 CPU 上内存，是换页的标准内存。
  // cudaHostAlloc 分配 CPU 上内存，是锁定页面的内存，也称之为 Pinned 内存。
  // 操作系统会保证此类内存总是在物理内存中，而不会换页到磁盘上。
  // 此属性帮助 GPU 通过 直接内存访问 DMA 将数据复制，而不需 CPU 干预。
  cudaHostAlloc((void**)&host_a, N * 2 * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&host_b, N * 2 * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&host_c, N * 2 * sizeof(int), cudaHostAllocDefault);

  // allocate the device memory.
  cudaMalloc((void**)&device_a_stream0, N * sizeof(int));
  cudaMalloc((void**)&device_b_stream0, N * sizeof(int));
  cudaMalloc((void**)&device_c_stream0, N * sizeof(int));
  cudaMalloc((void**)&device_a_stream1, N * sizeof(int));
  cudaMalloc((void**)&device_b_stream1, N * sizeof(int));
  cudaMalloc((void**)&device_c_stream1, N * sizeof(int));

  // Initialize.
  for (int i = 0; i < N * 2; ++i)
  {
    host_a[i] = 2 * i * i;
    host_b[i] = i;
  }

  // cudaMemcpyAsync CUDA API 异步传输操作。
  cudaMemcpyAsync(device_a_stream0, host_a, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
  cudaMemcpyAsync(device_a_stream1, host_a + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(device_b_stream0, host_b, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
  cudaMemcpyAsync(device_b_stream1, host_b + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

  // Call kernel passing device pointers as parameters.
  gpuAdd << <512, 512, 0, stream0 >> > (device_a_stream0, device_b_stream0, device_c_stream0);
  gpuAdd << <512, 512, 0, stream1 >> > (device_a_stream1, device_b_stream1, device_c_stream1);

  // Copy result back to host memory from device memory.
  cudaMemcpyAsync(host_c, device_c_stream0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
  cudaMemcpyAsync(host_c + N, device_c_stream1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

  // sync 同步。
  cudaDeviceSynchronize();
	cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);
  // CUDA Events.
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
  printf("Time to add %d numbers: %3.1f ms\n",2 * N, elapsedTime);
  
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

  // Free up device memory
	cudaFree(device_a_stream0);
	cudaFree(device_b_stream0);
	cudaFree(device_c_stream0);
	cudaFree(device_a_stream1);
	cudaFree(device_b_stream1);
  cudaFree(device_c_stream1);
  
  // Free up host memory.
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

  return 0;
}