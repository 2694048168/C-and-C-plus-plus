#include <iostream>
#include <memory>
#include <cuda_runtime.h>

/** 设备属性
 * 了解 GPU 硬件设备的各项属性，有利于高效的利用 GPU 资源。
 * 针对特定的硬件设备 GPU，为应用程序开发分配合理的资源。
 * 具体使用的时候，需要查 CUDA 编程手册以及对应 GPU 硬件资源信息。
 *
 * D:\Nvidia\Samples\1_Utilities\deviceQuery\deviceQuery.cpp
 */
int main(int argc, char **argv)
{
  int device_count = 0; // 可用设备数量
  int device; // 设备名称编号
  int driver_version; // CUDA 驱动版本号
  int runtime_version; // 运行时引擎版本

  // 通过查询 cudaDeviceProp 结构体来找到每个设备的相关信息，该结构体返回所有设备的属性。
  // 如果有多个可用设备，可使用 for 循环遍历所有设备属性。
  // get count of the avaiable CUDA hardware device in system.
  cudaGetDeviceCount(&device_count);

  // This function returns count of number of CUDA enable devices and 0 if there are no CUDA capable devices.
  if (device_count == 0)
  {
    // printf("There are no available devices that support CUDA.\n");
    std::cout << "There are no available devices that support CUDA.\n";
  }
  else
  {
    // printf("Detected %d CUDA Capable devices.\n", device_count);
    std::cout << "Detected " << device_count << " CUDA Capable devices.\n";
  }

  /**通用设备信息
   * cudaDeviceProp 结构体提供可以用来识别设备以及确认使用的版本信息的属性，name属性以字符串形式返回设备名称。
   * cudaDriverGetVersion 获取设备使用的 CUDA Driver。
   * cudaRuntimeGetVersion 获取设备运行时引擎的版本。
   * clockRate属性获取 GPU 的时钟速率。
   * multiProcessorCount属性用于判断设备上流多处理器的个数。
   */
  // 获取设备名称编号
  cudaGetDevice(&device);
  // printf("ID of device: %d\n", device);
  std::cout << "ID of device: " << device << std::endl;

  // 获取设备结构体信息
  cudaDeviceProp device_property;
  cudaGetDeviceProperties(&device_property, device);
  // printf("\nDevice %s: \n", device_property.name);
  // device_property.name 获取 GPU 型号
  std::cout << "The type of hardware GPU is: " << device_property.name << std::endl;

  // 获取 CUDA 驱动版本号
  cudaDriverGetVersion(&driver_version);
  std::cout << "CUDA Driver Version is: CUDA " << driver_version / 1000 << "." << (driver_version % 100) / 10 << std::endl;

  // 获取运行时引擎版本
  cudaRuntimeGetVersion(&runtime_version);
  std::cout << "CUDA Runtime Driver Version is: CUDA " << driver_version / 1000 << "." << (driver_version % 100) / 10 << std::endl;

  // GPU 显存容量
  // printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n", (float)device_property.totalGlobalMem / 1048576.0f, 
                                                                      // (unsigned long long)device_property.totalGlobalMem);
  std::cout << "Total amount of global memory: " << (float)device_property.totalGlobalMem / 1048576.0f << " MBytes" << std::endl;

  // 具有最多流处理器的设备,如果有多个设备
  printf(" (%2d) mutilprocessors\n", device_property.multiProcessorCount);
  // std::cout << device_property.mutilProcessorCount << "mutilprocessors" << std::endl;

  // GPU 时钟速率，以 KHz 为单位进行返回
  // printf("GPU max clock rate: %.0f MHz (%.2f GHz)\n", device_property.clockRate * 1e-3f, device_property.clockRate * 1e-6f);
  std::cout << "GPU max clock rate: " << device_property.clockRate * 1e-6f << " GHz" << std::endl;

  // 显存频率
  // printf("Memory clock rate: %.0f MHz\n", device_property.memoryClockRate * 1e-3f);
  std::cout << "Memory clock rate: " << device_property.memoryClockRate * 1e-3f << " MHz" << std::endl;

  // 显存位宽
  // printf("Memory Bus Width: %d-bit\n", device_property.memoryBusWidth);
  std::cout << "Memory Bus Width: " << device_property.memoryBusWidth << "-bit" << std::endl;

  // L2 缓存
  if (device_property.l2CacheSize)
  {
    // printf("L2 Cache size: %d bytes\n", device_property.l2CacheSize);
    std::cout << "L2 Cache size: " << device_property.l2CacheSize << " bytes" << std::endl;
  }

  // 常量内存
  // printf("Toal amount of constant memory: %lu bytes\n", device_property.totalConstMem);
  std::cout << "Toal amount of constant memory: " << device_property.totalConstMem << " bytes" << std::endl;

  // 共享内存
  // printf("Toal amount of shared memory per block: %lu bytes\n", device_property.sharedMemPerBlock);
  std::cout << "Toal amount of shared memory per block: " << device_property.sharedMemPerBlock << " bytes" << std::endl;

  // 每一个块可用寄存器总数
  // printf("Toal amount of registers available per block: %d\n", device_property.regsPerBlock);
  std::cout << "Toal amount of registers available per block: " << device_property.regsPerBlock << std::endl;

  // 网格grid 块block 线程thread 可以时多维的，每一个维度中可用并行启动多少个线程和块
  // 这对于内核参数的配置十分重要
  // printf("Maximum number of threads per multiprocessor: %d\n", device_property.maxThreadsPerMutilProcessor);
  // std::cout << "Maximum number of threads per multiprocessor: " << device_property.maxThreadsPerMutilProcessor << std::endl;
  
  // printf("Maximum number of threads per block: %d\n", device_property.maxThreadsPerBlock);
  std::cout << "Maximum number of threads per block: " << device_property.maxThreadsPerBlock << std::endl;

  // printf("Max dimension size of a thread block (x, y, z): (%d, %d, %d)\n", 
  //                                              device_property.maxThreadsDim[0],
  //                                              device_property.maxThreadsDim[1],
  //                                              device_property.maxThreadsDim[2]);
  std::cout << "Max dimension size of a thread block (x, y, z): " << "(" << 
                                               device_property.maxThreadsDim[0] << "," <<
                                               device_property.maxThreadsDim[1] << "," <<
                                               device_property.maxThreadsDim[2] << ")" << std::endl;

  // printf("Max dimension size of a grid size (x, y, z): (%d, %d, %d)\n", 
  //                                           device_property.maxGridSize[0],
  //                                           device_property.maxGridSize[1],
  //                                           device_property.maxGridSize[2]);
  std::cout << "Max dimension size of a grid size (x, y, z): " << "(" << 
                                               device_property.maxGridSize[0] << "," <<
                                               device_property.maxGridSize[1] << "," <<
                                               device_property.maxGridSize[2] << ")" << std::endl;

  // 查看设备是否支持双精度浮点操作，并为应用程序设置该设备
  memset(&device_property, 0, sizeof(cudaDeviceProp));
  // if major > 1 and minor > 3, then the device supports double precision.
  device_property.major = 1;
  device_property.minor = 3;

  // 选择特定属性的设备
  cudaChooseDevice(&device, &device_property);

  // printf("ID of device which supports double precision is: %d\n", device);
  std::cout << "ID of device which supports double precision is: " << device << std::endl;
  // 设置设备为应用程序所用设备
  cudaSetDevice(device);

  return 0;
}