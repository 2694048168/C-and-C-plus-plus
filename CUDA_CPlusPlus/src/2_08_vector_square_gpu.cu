#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/**并行通信模式
 * 0、Map模式，映射，该通信模式中，每个线程或者任务读取单一输入，产生一个输出，一对一的操作。
 * 1、Gather模式，收集，该通信模式中，每个线程或者任务具有多个输入，产生单一输出，多对一的操作。
 * 2、Scatter模式，分散式，该通信模式中，每个线程或者任务读取单一输入，向存储器产生多个输出，一对多的操作。
 * 3、Stencil模式，蒙板，该通信模式中，线程读取固定形状的相邻元素，图像卷积中卷积核。
 * 4、Transpose模式，转置，该通信模式中，输入矩阵为行主序，需要输出矩阵为列主序。
 *
 * CUDA 编程所遵循的模式，使用官网提供的类型应用编程模式，利用范例语法模式很有用。
 */

// 注意启动内核时参数的配置，块的数量和线程的数量控制，不能超过GPU硬件本身的限制。
// 应该合理的选择合适数量的块和每一个块所用的线程数。
// Defining number of elements in vector.
#define N 100

// Defining kernel function for squaring number.
__global__ void gpu_vector_square(double *device_in, double *device_out)
{
  // Getting thread index for current kernel.
  int tid = threadIdx.x; // handle the data at this index.
  double temp = device_in[tid];
  device_out[tid] = temp * temp;
}

int main(int argc, char **argv)
{
  // Defining vector for host.
  double host_in[N], host_out[N];
  double *device_in, *device_out;

  // Allocate the memory on the device GPU.
  cudaMalloc((void**)&device_in, N * sizeof(double));
  cudaMalloc((void**)&device_out, N * sizeof(double));

  // Initializing vector.
  for (unsigned int i = 0; i < N; ++i)
  {
    host_in[i] = i;
  }

  // Copy vector from host to device.
  cudaMemcpy(device_in, host_in, N * sizeof(double), cudaMemcpyHostToDevice);

  // Calling kernel with one block and N threads per block.
  gpu_vector_square << <1, N>> > (device_in, device_out);

  // Coping result back to host from device memory.
  cudaMemcpy(host_out, device_out, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Printing result on console.
  std::cout << "Square of number on GPU." << std::endl;
  for (unsigned int i = 0; i < N; ++i)
  {
    printf("The square of %f is %f\n", host_in[i], host_out[i]);
  }

  // Free up memory.
  cudaFree(device_in);
  cudaFree(device_out);

  return 0;
}