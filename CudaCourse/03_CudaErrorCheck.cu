/**
 * @file 03_CudaErrorCheck.cu
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-05-23
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstdio>
#include <cstdlib>

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

__host__ void setGPU() {
  // detect the number of GPU on the computer.
  int deviceNum = 0;
  cudaError_t error =
      ErrorCheck(cudaGetDeviceCount(&deviceNum), __FILE__, __LINE__);

  if (error != cudaSuccess || deviceNum == 0) {
    printf("No CUDA campatable GPU found!\n");
    exit(-1);
  } else {
    printf("The number of GPUs: %d\n", deviceNum);
  }

  // set the current execuate GPU.
  int deviceID = 0;
  error = ErrorCheck(cudaSetDevice(deviceID), __FILE__, __LINE__);
  if (error != cudaSuccess) {
    printf("Fail to set GPU-ID %d for current computing\n", deviceID);
  } else {
    printf("Set GPU-ID %d for current computing\n", deviceID);
  }
}

// device function
__device__ float add(const float x, const float y) { return x + y; }

// kernel function
__global__ void addMatrix_GPU(float *mat_A, float *mat_B, float *mat_C,
                              const int NUM) {
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int idx = block_id * blockDim.x + thread_id;

  // 防止并行 cuda-stream-processor 与 task-number 不能整除
  if (idx >= NUM)
    return;
  mat_C[idx] = add(mat_A[idx], mat_B[idx]);
}

// host function code
void init_data(float *addr, int elem_count) {
  for (int idx{0}; idx < elem_count; ++idx) {
    addr[idx] = (float)(rand() & 0xFF) / 10.f;
  }
}

// ------------------------------------
int main(int argc, const char **argv) {

  // ====Step1. set the GPU device
  setGPU();

  // ====Step2. malloc the memory on host, and init-memory
  const size_t element_NUM = 4096;
  const size_t byte_count = element_NUM * sizeof(float);

  float *fpHost_matA;
  float *fpHost_matB;
  float *fpHost_matC;

  fpHost_matA = (float *)malloc(byte_count);
  fpHost_matB = (float *)malloc(byte_count);
  fpHost_matC = (float *)malloc(byte_count);
  if (fpHost_matA != nullptr && fpHost_matB != nullptr &&
      fpHost_matC != nullptr) {
    memset(fpHost_matA, 0, byte_count);
    memset(fpHost_matB, 0, byte_count);
    memset(fpHost_matC, 0, byte_count);
  } else {
    printf("Fail to allocate Host memory\n");
    exit(-1);
  }

  // ====Step3. malloc the memory on device, and init-memory
  float *fpDevice_matA;
  float *fpDevice_matB;
  float *fpDevice_matC;

  cudaMalloc((float **)&fpDevice_matA, byte_count);
  cudaMalloc((float **)&fpDevice_matB, byte_count);
  cudaMalloc((float **)&fpDevice_matC, byte_count);
  if (fpDevice_matA != nullptr && fpDevice_matB != nullptr &&
      fpDevice_matC != nullptr) {
    cudaMemset(fpDevice_matA, 0, byte_count);
    cudaMemset(fpDevice_matB, 0, byte_count);
    cudaMemset(fpDevice_matC, 0, byte_count);
  } else {
    printf("Fail to allocate Host memory\n");

    free(fpHost_matA);
    free(fpHost_matB);
    free(fpHost_matC);

    exit(-1);
  }

  // ====Step. init the data on host
  srand(42);
  init_data(fpHost_matA, element_NUM);
  init_data(fpHost_matB, element_NUM);

  // ====Step4. Copy data from Host into Device
  cudaMemcpy(fpDevice_matA, fpHost_matA, byte_count, cudaMemcpyHostToDevice);
  cudaMemcpy(fpDevice_matB, fpHost_matB, byte_count, cudaMemcpyHostToDevice);
  cudaMemcpy(fpDevice_matC, fpHost_matC, byte_count, cudaMemcpyHostToDevice);

  // ====Step5. kernel function execuate on Device
  const int block_dim = 2048; //! Error: max-thread==1024
  dim3 block(block_dim);
  dim3 grid(element_NUM / block_dim);
  addMatrix_GPU<<<grid, block>>>(fpDevice_matA, fpDevice_matB, fpDevice_matC,
                                 element_NUM);

  // 捕捉核函数错误方法, 在调用核函数后，追加如下代码:
  // 第一条语句作用是捕捉第二条同步函数之前的最后一个错误;
  // 第二条语句同步主机与设备，因为CPU和GPU是异构架构
  ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
  ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

  //   cudaDeviceSynchronize();
  //   异构计算 Heterogeneous computing

  // ====Step6. Host get result from Device
  // cudaMemcpy function will wait kernel function over(阻塞/同步)
  cudaMemcpy(fpHost_matC, fpDevice_matC, byte_count, cudaMemcpyDeviceToHost);

  for (int idx{0}; idx < 10; ++idx) {
    printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", idx + 1,
           fpHost_matA[idx], fpHost_matB[idx], fpHost_matC[idx]);
  }

  // ====Step7. free and cudaFree memory
  if (fpHost_matA) {
    free(fpHost_matA);
    fpHost_matA = nullptr;
  }
  if (fpHost_matB) {
    free(fpHost_matB);
    fpHost_matB = nullptr;
  }
  if (fpHost_matC) {
    free(fpHost_matC);
    fpHost_matC = nullptr;
  }

  if (fpDevice_matA) {
    cudaFree(fpDevice_matA);
    fpDevice_matA = nullptr;
  }
  if (fpDevice_matB) {
    cudaFree(fpDevice_matB);
    fpDevice_matB = nullptr;
  }
  if (fpDevice_matC) {
    cudaFree(fpDevice_matC);
    fpDevice_matC = nullptr;
  }

  // ====Step8. reset device
  cudaDeviceReset();
  printf("Matrix-add via Heterogeneous computing successfully\n");
  return 0;
}
