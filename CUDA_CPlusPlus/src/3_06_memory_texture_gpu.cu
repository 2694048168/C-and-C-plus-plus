#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 10
#define N 10

// Define texture reference for 1-d access.
// texture <> CUDA 内置类型变量。
// 具体参数列表以及使用例子，可以查看 CUDA 编程手册。
texture <float, 1, cudaReadModeElementType> textureRef;

// Kernel function for using texture memory.
__global__ void gpu_texture_memory(int n, float *device_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) 
  {
		float temp = tex1D(textureRef, float(idx));
		device_out[idx] = temp;
	}
}

int main(int argc, char *argv[])
{
	// Calculate number of blocks to launch.
  int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
  
	// Declare device pointer.
  float *device_out;
  
	// allocate space on the device for the result.
  cudaMalloc((void**)&device_out, sizeof(float) * N);
  
	// allocate space on the host for the results
  float *host_out = (float*)malloc(sizeof(float)*N);
  
	// Declare and initialize host array.
	float host_in[N];
  for (unsigned int i = 0; i < N; ++i) 
  {
		host_in[i] = float(i);
  }
  
  // Define CUDA Array.
  // CUDA 数组，cudaArray CUDA 内置的数据类型。
  // 具体使用方法查看 CUDA 编程手册。
	cudaArray *cu_Array;
  cudaMallocArray(&cu_Array, &textureRef.channelDesc, N, 1);
  
	// Copy data to CUDA Array.
	cudaMemcpyToArray(cu_Array, 0, 0, host_in, sizeof(float)*N, cudaMemcpyHostToDevice);
	
	// bind a texture to the CUDA array.
	cudaBindTextureToArray(textureRef, cu_Array);
	// Call Kernel.	
  gpu_texture_memory <<<num_blocks, NUM_THREADS >>> (N, device_out);
	
	// copy result back to host.
  cudaMemcpy(host_out, device_out, sizeof(float)*N, cudaMemcpyDeviceToHost);
  
	printf("Use of Texture memory on GPU: \n");
  for (unsigned int i = 0; i < N; ++i) 
  {
		printf("Texture element at %d is : %f\n",i, host_out[i]);
  }
  
  // Free dynamically managed memory.
	free(host_out);
	cudaFree(device_out);
	cudaFreeArray(cu_Array);
  cudaUnbindTexture(textureRef);
  
  return 0;
}
