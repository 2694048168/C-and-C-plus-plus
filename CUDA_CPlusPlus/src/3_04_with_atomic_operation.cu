/**atomic operation 原子操作
 * 考虑大量的线程需要同时访问同一内存区域的内存，特别进行写入操作，容易出现很危险的情况。
 * 原子操作是不可以被其他线程扰乱的原子性的整体完成的一组操作。
 * 《UNIX 环境高级编程》书籍中有对 原子操作 详细的讲解。
 */

#include <stdio.h>

// Define the number of threads.
#define NUM_THREADS 10000
// Define the size of vector.
#define SIZE 10
// Define the number of blocks.
#define BLOCK_WIDTH 100
 
__global__ void gpu_increment_atomic(int *device_a)
{
  // Calculate thread index.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread increments elements which wraps at SIZE.
  tid = tid % SIZE;
  // CUDA API atomicAdd is an atomic operation.
  atomicAdd(&device_a[tid], 1);
}

int main(int argc, char *argv[])
{
  printf("%d total threads in %d blocks writing into %d array elements\n", NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

	// declare and allocate host memory.
	int host_a[SIZE];
	const int ARRAY_BYTES = SIZE * sizeof(int);

	// declare and allocate GPU memory.
	int * device_a;
	cudaMalloc((void **)&device_a, ARRAY_BYTES);
  // Initialize GPU memory to zero.
  // CUDA API 初始化显存上的工作。
	cudaMemset((void *)device_a, 0, ARRAY_BYTES);
  
  // kernel call.
	gpu_increment_atomic <<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >>> (device_a);
	

	// copy back the array to host memory.
	cudaMemcpy(host_a, device_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	printf("Number of times a particular Array index has been incremented is: \n");
	for (int i = 0; i < SIZE; ++i) 
	{ 
		printf("index: %d --> %d times\n ", i, host_a[i]); 
	}
	// Free device pointers.
	cudaFree(device_a);

  return 0;
}
 