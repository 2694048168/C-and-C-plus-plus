#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining two constants.
// 使用 __constant__ 修饰限定符来声明变量存储于常量存储器中。
__constant__ int constant_f;
__constant__ int constant_g;

#define N	5

// Kernel function for using constant memory.
__global__ void gpu_constant_memory(float *device_in, float *device_out) 
{
	//Thread index for current kernel.
	int tid = threadIdx.x;	
	device_out[tid] = constant_f * device_in[tid] + constant_g;
}

int main(int argc, char *argv[]) 
{
	// Defining Arrays for host.
	float host_in[N], host_out[N];
  
  // Defining Pointers for device.
	float *device_in, *device_out;
	int h_f = 2;
  int h_g = 20;
  
	// allocate the memory on the device GPU.
	cudaMalloc((void**)&device_in, N * sizeof(float));
	cudaMalloc((void**)&device_out, N * sizeof(float));
  
  // Initializing Array
  for (unsigned int i = 0; i < N; ++i) 
  {
		host_in[i] = i;
  }
  
	// Copy array data from host to device.
  cudaMemcpy(device_in, host_in, N * sizeof(float), cudaMemcpyHostToDevice);
  
  // Copy constants to constant memory.
  // CUDA API cudaMemcpyToSymbol 将常量复制到内核执行所需要的常量内存中。
  // 具体参数列表以及使用例子，可以查看 CUDA 编程手册。
	cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

	// Calling kernel with one block and N threads per block.
  gpu_constant_memory <<<1, N >>> (device_in, device_out);
  
	// Coping result back to host from device memory.
  cudaMemcpy(host_out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  
	// Printing result on console.
	printf("Use of Constant memory on GPU \n");
  for (unsigned int i = 0; i < N; ++i) 
  {
		printf("The expression for input %f is %f\n", host_in[i], host_out[i]);
  }
  
	//Free up memory
	cudaFree(device_in);
	cudaFree(device_out);
  
  return 0;
}
