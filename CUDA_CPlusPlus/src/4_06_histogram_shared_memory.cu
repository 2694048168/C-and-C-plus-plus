#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1000
#define NUM_BIN 16

__global__ void histogram_shared_memory(int *device_b, int *device_a)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = blockDim.x * gridDim.x;
	__shared__ int cache[256];
	cache[threadIdx.x] = 0;
	__syncthreads();
	
	while (tid < SIZE)
	{
		atomicAdd(&(cache[device_a[tid]]), 1);
		tid += offset;
	}
	__syncthreads();
	atomicAdd(&(device_b[threadIdx.x]), cache[threadIdx.x]);
}


int main(int argc, char **argv)
{
	// generate the input array on the host.
	int host_a[SIZE];
  for (int i = 0; i < SIZE; ++i) 
  {
		//host_a[i] = bit_reverse(i, log2(SIZE));
		host_a[i] = i % NUM_BIN;
  }
  
	int host_b[NUM_BIN];
  for (int i = 0; i < NUM_BIN; ++i) 
  {
		host_b[i] = 0;
	}

	// declare GPU memory pointers
	int *device_a, *device_b;

	// allocate GPU memory
	cudaMalloc((void **)&device_a, SIZE * sizeof(int));
	cudaMalloc((void **)&device_b, NUM_BIN * sizeof(int));

	// transfer the arrays to the GPU
	cudaMemcpy(device_a, host_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

	// launch the kernel
	histogram_shared_memory <<<SIZE / 256, 256 >>> (device_b, device_a);

	// copy back the result from GPU
	cudaMemcpy(host_b, device_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Histogram using 16 bin is: \n");
  for (int i = 0; i < NUM_BIN; ++i) 
  {
		printf("bin %d: count %d\n", i, host_b[i]);
	}

	// free GPU memory allocation
	cudaFree(device_a);
	cudaFree(device_b);

	return 0;
}