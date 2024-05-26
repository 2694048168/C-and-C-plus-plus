/**
 * @file 05_QueryDeviceInfo.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-25
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdio>

cudaError_t ErrorCheck(cudaError_t error_code, const char *filename,
                       int lineNumber) {
  if (error_code != cudaSuccess) {
    printf(
        "\n[====CUDA Error]:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, "
        "line%d\r\n\n",
        error_code, cudaGetErrorName(error_code),
        cudaGetErrorString(error_code), filename, lineNumber);
    return error_code;
  }
  return error_code;
}

//  查询GPU计算核心数量
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
  case 2: // Fermi
    if (devProp.minor == 1)
      cores = mp * 48;
    else
      cores = mp * 32;
    break;
  case 3: // Kepler
    cores = mp * 192;
    break;
  case 5: // Maxwell
    cores = mp * 128;
    break;
  case 6: // Pascal
    if ((devProp.minor == 1) || (devProp.minor == 2))
      cores = mp * 128;
    else if (devProp.minor == 0)
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 7: // Volta and Turing
    if ((devProp.minor == 0) || (devProp.minor == 5))
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 8: // Ampere
    if (devProp.minor == 0)
      cores = mp * 64;
    else if (devProp.minor == 6)
      cores = mp * 128;
    else if (devProp.minor == 9)
      cores = mp * 128; // ada lovelace
    else
      printf("Unknown device type\n");
    break;
  case 9: // Hopper
    if (devProp.minor == 0)
      cores = mp * 128;
    else
      printf("Unknown device type\n");
    break;
  default:
    printf("Unknown device type\n");
    break;
  }
  return cores;
}

// -----------------------------------
int main(int argc, const char **argv) {
  int device_id = 0;
  ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

  cudaDeviceProp device_prop;
  ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__,
             __LINE__);

  printf("\nDevice id:                                 %d\n", device_id);
  printf("Device name:                               %s\n", device_prop.name);
  printf("Compute capability:                        %d.%d\n",
         device_prop.major, device_prop.minor);
  printf("Amount of global memory:                   %g GB\n",
         device_prop.totalGlobalMem / (1024.0 * 1024 * 1024));
  printf("Amount of constant memory:                 %g KB\n",
         device_prop.totalConstMem / 1024.0);
  printf("Maximum grid size:                         %d %d %d\n",
         device_prop.maxGridSize[0], device_prop.maxGridSize[1],
         device_prop.maxGridSize[2]);
  printf("Maximum block size:                        %d %d %d\n",
         device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
         device_prop.maxThreadsDim[2]);
  printf("Number of SMs:                             %d\n",
         device_prop.multiProcessorCount);
  printf("Maximum amount of shared memory per block: %g KB\n",
         device_prop.sharedMemPerBlock / 1024.0);
  printf("Maximum amount of shared memory per SM:    %g KB\n",
         device_prop.sharedMemPerMultiprocessor / 1024.0);
  printf("Maximum number of registers per block:     %d K\n",
         device_prop.regsPerBlock / 1024);
  printf("Maximum number of registers per SM:        %d K\n",
         device_prop.regsPerMultiprocessor / 1024);
  printf("Maximum number of threads per block:       %d\n",
         device_prop.maxThreadsPerBlock);
  printf("Maximum number of threads per SM:          %d\n\n",
         device_prop.maxThreadsPerMultiProcessor);

  //  查询GPU计算核心数量
  printf("[====]Compute cores is %d.\n", getSPcores(device_prop));

  return 0;
}
