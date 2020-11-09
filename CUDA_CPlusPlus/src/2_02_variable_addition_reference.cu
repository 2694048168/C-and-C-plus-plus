#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel function to add two variables, parametere are passed by reference.
__global__ void gpu_add(int *device_a, int *device_b, int *device_c)
{
  *device_c = *device_a + *device_b;
}

int main(int argc, char **argv)
{
  // Definition host variables and device pointer variables.
  int host_a, host_b, host_c;
  int *device_a, *device_b, *device_c;

  // Initializing host vaiables.
  host_a = 1;
  host_b = 4;

  // Allocating memory for device pointers.
  cudaMalloc((void**)&device_a, sizeof(int));
  cudaMalloc((void**)&device_b, sizeof(int));
  cudaMalloc((void**)&device_c, sizeof(int));

  // Coping value of host variables in device memory.
  cudaMemcpy(device_a, &host_a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, &host_b, sizeof(int), cudaMemcpyHostToDevice);

  // 按引用传递参数，高效，注意数据所在内存位置，主机还是设备，两者之间的内存拷贝，数据交换。
  // Calling kernel with one thread and one block with parameters passed by reference.
  gpu_add << <1, 1>> > (device_a, device_b, device_c);

  // Coping result from device memory to host.
  cudaMemcpy(&host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);

  // printf("Passing parameter by reference output: d% + d% = d%\n", host_a, host_b, host_c);
  std::cout << "Passing parameter by reference output: " << host_a << "+ " << host_b << " = " << host_c << std::endl;

  // Free up memory.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}