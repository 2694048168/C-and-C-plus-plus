#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/** CUDA API
 * __global__ CUDA C 关键字限定符，表示函数被声明为设备函数，该函数只能从主机上调用，并在设备上执行。
 * __device__ CUDA C 关键字限定符，表示函数被声明为设备函数，该函数只能从设备上调用，并在设备上执行。
 * __host__ CUDA C 关键字限定符，表示函数被声明为主机函数，该函数只能从其他主机上调用，并在主机上执行。
 */
// Definition of kernel function to add two variables.
__global__ void gpu_add(int device_a, int device_b, int *device_c)
{
  *device_c = device_a + device_b;
}

int main(int argc, char **argv)
{
  // Definition host variable to store answer.
  int host_c;

  // Definition device pointer
  int *device_c;
  
  /** CUDA API 
   * cudaMalloc 类似 C 中动态内存分配函数 Malloc 函数，该函数用于在设备上分配特定大小的内存。
   * cudaMalloc(void * *device_pointer, size_t size)
   * Example: cudaMalloc((void**)&device_c, sizeof(int));
   * 返回指向分配内存首地址的指针。
   */
  // Allocating memory for device pointer.
  cudaMalloc((void**)&device_c, sizeof(int));

  /**kernel call configure
   * kernel << <number of blocks, number of threads per block, size of shared memory>> > (parameters for kernel)
   * 启动内核时，块的数量和每一块的线程数量的配置非常重要，GPU资源的精准控制和高效使用。
   */
  // Kernel call by passing 1 and 4 as inputs and storing answer in device_c.
  // << <1, 1>> > means 1 block is executed with 1 thread per block.
  // 按值传递参数列表，不建议按值传递参数列表。
  gpu_add << <1, 1>> > (1, 4, device_c);

  /** CUDA API 
   * cudaMemcpy 类似 C 中内存拷贝函数 Memcpy 函数，该函数用于将内存区域复制到主机或者设备上的其他区域。
   * cudaMemcpy(void * dst_ptr, const void * src_ptr, size_t size, enum cudaMemcpyKind kind)
   * Example: cudaMemcpy(&host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
   * 第一个参数是目标指针；第二个参数是源指针；第三个参数是数据复制大小；第四个参数是数据复制的方向；
   * 前两个指针参数必须与数据复制的方向一致，两两组合，四个枚举值：
   * cudaMemcpyDeviceToHost；cudaMemcpyHostToDevice；cudaMemcpyDeviceToDevice；cudaMemcpyHostToHost
   */
  // Copy result from device memory to host memory.
  cudaMemcpy(&host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);

  // printf("1 + 4 = %d\n", host_c);
  std::cout << "1 + 4 = " << host_c << std::endl;

  /** CUDA API 
   * cudaFree 类似 C 中内存释放函数 free 函数，该函数用于将动态分配的内存释放。
   * cudaFree(void * device_ptr)
   * Example: cudaFree(device_c);
   */
  // Free up memory.
  cudaFree(device_c);

  return 0;
}