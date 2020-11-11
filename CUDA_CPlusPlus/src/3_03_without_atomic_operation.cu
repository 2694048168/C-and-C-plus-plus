/**atomic operation 原子操作
 * 考虑大量的线程需要同时访问同一内存区域的内存，特别进行写入操作，容易出现很危险的情况。
 * 原子操作是不可以被其他线程扰乱的原子性的整体完成的一组操作。
 * 《UNIX 环境高级编程》书籍中有对 原子操作 详细的讲解。
 * 如果没有这样 原子操作 ，则会出现一些未知的不可控制的情况出现。
 */

#include <stdio.h>

#define NUM_THREADS 10000
#define SIZE  10
#define BLOCK_WIDTH 100
 
__global__ void gpu_increment_without_atomic(int *d_a)
{
  // Calculate thread id for current thread
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread increments elements wrapping at SIZE variable
  tid = tid % SIZE;
  d_a[tid] += 1;
}
 
int main(int argc, char **argv)
{
  printf("%d total threads in %d blocks writing into %d array elements\n", NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);
 
  // declare and allocate host memory
  int h_a[SIZE];
  const int ARRAY_BYTES = SIZE * sizeof(int);

  // declare and allocate GPU memory
  int * d_a;
  cudaMalloc((void **)&d_a, ARRAY_BYTES);
  //Initialize GPU memory to zero
  cudaMemset((void *)d_a, 0, ARRAY_BYTES);

  gpu_increment_without_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);

  // copy back the array to host memory
  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  printf("Number of times a particular Array index has been incremented without atomic add is: \n");
  for (int i = 0; i < SIZE; i++)
  {
    printf("index: %d --> %d times\n ", i, h_a[i]);
  }

  cudaFree(d_a);

  return 0;
}