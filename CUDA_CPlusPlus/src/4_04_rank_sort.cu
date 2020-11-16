#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define arraySize 6
#define threadPerBlock 6

/**枚举排序或者秩排序算法
 * 对于数组中的每一个元素，通过统计小于其值的数组中其他元素的数量，
 * 该统计数量就是该元素在最终结果数组中的位置索引。
 */
// Define kernel function to sort array with rank.
__global__ void rank_sort_kernel(int *device_a, int *device_b)
{
  unsigned int count = 0;
  unsigned int tid = threadIdx.x;
  unsigned int ttid = blockIdx.x * threadPerBlock + tid;
  int val = device_a[ttid];

  // using shared memory.
  __shared__ int cache[threadPerBlock];
  for (unsigned int i = tid; i < arraySize; i += threadPerBlock)
  {
    cache[tid] = device_a[i];
    __syncthreads();
    for (unsigned j = 0; j < threadPerBlock; ++j)
    {
      if (val > cache[j])
      {
        count++;
      }
      __syncthreads();
    }
  }
  device_b[count] = val;
}

int main(int argc, char **argv)
{
  int host_a[arraySize] = {5, 9, 2, 3, 8, 4};
  int host_b[arraySize];
  int *device_a, *device_b;

  cudaMalloc((void**)&device_a, arraySize * sizeof(int));
  cudaMalloc((void**)&device_b, arraySize * sizeof(int));

  cudaMemcpy(device_a, host_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

  rank_sort_kernel <<< arraySize / threadPerBlock, threadPerBlock >>> (device_a, device_b);

  cudaDeviceSynchronize();

  cudaMemcpy(host_b, device_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

  printf("The before sorted Array is: \n");
  for (unsigned int k = 0; k < arraySize; ++k) 
  {
		printf("%d\t", host_a[k]);
  }

  printf("\n\nThe Enumeration sorted Array is: \n");
  for (unsigned int k = 0; k < arraySize; ++k) 
  {
		printf("%d\t", host_b[k]);
  }
  
  cudaFree(device_a);
  cudaFree(device_b);

  return 0;
}