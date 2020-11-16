#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

// Define kernel function.
__global__ void gpuAdd(int *device_a, int *device_b, int *device_c)
{
  *device_c = *device_a + *device_b;
}

int main(int argc, char **argv)
{
  // Define host variables and device pointers.
  int host_a, host_b, host_c;
  int *device_a, *device_b, *device_c;
  
  // Initialize host variables.
  host_a = 11;
  host_b = 13;

  // CUDA 错误处理。
  cudaError_t cudaStatus;

  // Allocate GPU buffers for three vectors(two input, ont output).
  cudaStatus = cudaMalloc((void**)&device_c, sizeof(int));
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc failed.\n");
    // 跳转到标号 Error
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&device_a, sizeof(int));
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc failed.\n");
    // 跳转到标号 Error
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&device_b, sizeof(int));
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc failed.\n");
    // 跳转到标号 Error
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(device_a, &host_a, sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpy failed.\n");
    // 跳转到标号 Error
    goto Error;
  }
  cudaStatus = cudaMemcpy(device_b, &host_b, sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpy failed.\n");
    // 跳转到标号 Error
    goto Error;
  }

  // Launch kernel on GPU device with one thread for each element.
  gpuAdd <<< 1, 1 >>> (device_a, device_b, device_c);

  // Check for any errors launching the kernel.
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    // 跳转到标号 Error
    goto Error;
  }

  // Copy output vector from GPU device to host memory.
  cudaStatus = cudaMemcpy(&host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpy failed.\n");
    // 跳转到标号 Error
    goto Error;
  }

  printf("Passing parameters by reference output: %d + %d = %d\n", host_a, host_b, host_c);

Error:
  // 标号处理，直接释放内存，结束程序。
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}