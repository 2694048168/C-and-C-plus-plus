#include <iostream>
#include <stdio.h>

/**并行启动的块和线程随机顺序执行
 * 在配置内核参数时，可以指定并行启动的块和线程的数量，但是其执行顺序是随机的。
 * 每一个块和块内的每一个线程都有自己的ID号， CUDA C 内置变量 blockIdx.x 和 threadIdx.x 用于读取块ID 和 线程ID。
 */
__global__ void first_kernel(void)
{
  // blockIdx.x gives the block number of current kernel
  printf("Hello, I am thread in block: %d\n", blockIdx.x);
}

int main(int argc, char **argv)
{
  // A kernel call with 16 blocks and 1 thread per block.
  first_kernel << <16, 1>> > ();

  /** CUDA API 
   * 内核启动是一个异步操作，只要发布内核启动命令，不等内核执行完成，控制权就会立刻返回调用内核的主机CPU线程。
   * 使用 cudaDeviceSynchronize() 函数，内核的结果将通过标准输出显示，同时应用程序则会在内核执行完成之后才退出。
   */
  // Function used for waiting for all kernels to finish.
  cudaDeviceSynchronize();

  // printf("All threads are finished!\n");
  std::cout << "All thread are finished" << std::endl;

  return 0;
}