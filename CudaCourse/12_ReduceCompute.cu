/**
 * @file 12_ReduceCompute.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utility.cuh"
#include <iostream>

void initialData(float *addr, int elemCount) {
  for (int i = 0; i < elemCount; i++) {
    addr[i] = (float)(rand() & 0xFF) / 10.f;
  }
  return;
}

// GPU计算数组加法
__global__ void addFromGPU(float *A, float *B, float *C, const int N) {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id >= N)
    return;
  // C[id] = add(A[id], B[id]);
  C[id] = A[id] + B[id];
}

// CPU计算数组加法
void addFromCPU(float *A, float *B, float *C, const int N) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }

  return;
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
      break;
    }
  }

  if (match)
    printf("Arrays match.\n\n");
  else
    printf("Arrays do not match.\n\n");
}

// -------------------------------------
int main(int argc, const char **argv) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
  std::cout << "[INFO] 运行GPU设备: " << deviceProps.name << std::endl;
  std::cout << "[INFO] SM数量: " << deviceProps.multiProcessorCount
            << std::endl;
  std::cout << "[INFO] L2缓存大小: " << deviceProps.l2CacheSize / (1024 * 1024)
            << "M\n";
  std::cout << "[INFO] SM最大驻留线程数量: "
            << deviceProps.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "[INFO] 设备是否支持流优先级："
            << deviceProps.streamPrioritiesSupported << std::endl;
  std::cout << "[INFO] 设备是否支持在L1缓存中缓存全局内存: "
            << deviceProps.globalL1CacheSupported << std::endl;
  std::cout << "[INFO] 设备是否支持在L1缓存中缓存本地内存: "
            << deviceProps.localL1CacheSupported << std::endl;
  std::cout << "[INFO] 一个SM可用的最大共享内存量: "
            << deviceProps.sharedMemPerMultiprocessor / 1024 << "KB\n";
  std::cout << "[INFO] 一个SM可用的32位最大寄存器数量: "
            << deviceProps.regsPerMultiprocessor / 1024 << "K\n";
  std::cout << "[INFO] 一个SM最大驻留线程块数量: "
            << deviceProps.maxBlocksPerMultiProcessor << std::endl;
  std::cout << "[INFO] GPU内存带宽: " << deviceProps.memoryBusWidth
            << std::endl;
  std::cout << "[INFO] GPU内存频率: "
            << (float)deviceProps.memoryClockRate / (1024 * 1024) << "GHz\n\n";

  // --------------------------------------------------------------
  int iElemCount = 2048;                            // 设置元素数量
  size_t stBytesCount = iElemCount * sizeof(float); // 字节数

  // 1、分配主机内存
  float *fpHost_A = nullptr;
  float *fpHost_B = nullptr;
  float *fpHost_C = nullptr;
  float *fpDeviceRef = nullptr;
  fpHost_A = (float *)malloc(stBytesCount);
  fpHost_B = (float *)malloc(stBytesCount);
  fpHost_C = (float *)malloc(stBytesCount);
  fpDeviceRef = (float *)malloc(stBytesCount);
  srand(666); // 设置随机种子
  initialData(fpHost_A, iElemCount);
  initialData(fpHost_B, iElemCount);
  memset(fpHost_C, 0, stBytesCount);
  memset(fpDeviceRef, 0, stBytesCount);

  // 2、分配设备内存
  float *fpDevice_A = nullptr;
  float *fpDevice_B = nullptr;
  float *fpDevice_C = nullptr;
  CUDA_CHECK(cudaMalloc((float **)&fpDevice_A, stBytesCount));
  CUDA_CHECK(cudaMalloc((float **)&fpDevice_B, stBytesCount));
  CUDA_CHECK(cudaMalloc((float **)&fpDevice_C, stBytesCount));
  CUDA_CHECK(
      cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(fpDevice_C, 0, stBytesCount));

  // 3、CPU中进行计算
  addFromCPU(fpHost_A, fpHost_B, fpHost_C, iElemCount);
  // 4、GPU中进行计算
  dim3 block(64);
  dim3 grid((iElemCount + block.x - 1) / 64);
  addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C,
                              iElemCount); // 调用核函数
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(fpDeviceRef, fpDevice_C, stBytesCount,
                        cudaMemcpyDeviceToHost));

  // 对比CPU与GPU计算结果
  checkResult(fpHost_C, fpDeviceRef, iElemCount);

  for (int i = 0; i < 10; i++) // 打印
  {
    printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i + 1,
           fpHost_A[i], fpHost_B[i], fpDeviceRef[i]);
  }

  free(fpDeviceRef);
  free(fpHost_C);
  free(fpHost_B);
  free(fpHost_A);
  CUDA_CHECK(cudaFree(fpDevice_C));
  CUDA_CHECK(cudaFree(fpDevice_B));
  CUDA_CHECK(cudaFree(fpDevice_A));

  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
