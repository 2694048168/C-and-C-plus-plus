#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This defines size of a small square box or thread dimensions in one block.
// 分块矩阵计算，每一个块中线程将计算这些分块矩阵的元素，
// 进行矩阵乘法的总块数等于原始矩阵的大小除以分块的大小计算得到。
#define TILE_SIZE 2

// Define size of the square matrix.
const int size = 4;

// Define kernel funciotn for square matrix multiplication using non shared memory.
__global__ void gpu_matrix_multiplcation_nonshared(float *device_a, float *device_b, float *device_c, const int size)
{
  // Compute the index of rows and columns for matrix.
  // 将矩阵在存储器中按照 行主序 的方式线性存储，这样索引原始矩阵的元素计算方法：
  // index = 行号乘以矩阵的宽度 + 列号
  // 在图像处理中，常用一个图像的像素矩阵，按照 行主序 方式存储，
  // 然后从 行向量 转置为 列向量，作为神经网络的输入。
  unsigned int row, col;
  col = TILE_SIZE * blockIdx.x + threadIdx.x;
  row = TILE_SIZE * blockIdx.y + threadIdx.y;

  for (unsigned int k = 0; k < size; ++k)
  {
    // 第一个矩阵的行元素对应乘以第二个矩阵的列元素，计算的累和就得到结果矩阵的对应位置。
    // AB=C ——> sum (a[i,k] * b[k,j]) = c[i,j] when k form 1 to n.
    device_c[row * size + col] += device_a[row * size + k] * device_b[k * size + col];
  }
}

// Define kernel function for square matrix multiplication using shared memory.
__global__ void gpu_matrix_multiplcation_shared(float *device_a, float *device_b, float *device_c, const int size)
{
  unsigned int row, col;

  // 使用共享内存来存储计算分块矩阵，
  // 矩阵乘法中同样的数据被多次使用，这种情况正是共享内存的理想情况。
  __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

  // Calculate thread id.
  col = TILE_SIZE * blockIdx.x + threadIdx.x;
  row = TILE_SIZE * blockIdx.y + threadIdx.y;

  // 计算分块矩阵中的索引存储到共享内存中。
  // 保存需要重用的数据。
  for (unsigned int i = 0; i < size / TILE_SIZE; ++i)
  {
    shared_a[threadIdx.y][threadIdx.x] = device_a[row * size + (i * TILE_SIZE + threadIdx.x)];
    shared_b[threadIdx.y][threadIdx.x] = device_b[row * size + (i * TILE_SIZE + threadIdx.x)];
  }

  // 保证所有数据已经完成写入操作。
  __syncthreads();

  // 计算矩阵乘法。
  for (unsigned int j = 0; j < TILE_SIZE; ++j)
  {
    // 大量的计算，读取都发生在共享内存中，显著降低对全局内存的访问，提高性能。
    device_c[row * size + col] += shared_a[threadIdx.x][j] * shared_b[j][threadIdx.y];

    // synchronizing the threads.
    __syncthreads();
  }
}

// show the result of matrix multiplication.
/**使用1级指针访问二维数组
 * 因为数组本身在地址空间中就是连续排列的，根据行数和列数，
 * 计算出访问单元的 地址偏移量 就可以用一级指针遍历二维数组中的所有数据。
 */
// void show_result(float *ptr_array, int size)
// {
//   printf("The result of Matrix multiplication is: \n");
// 	for (int i = 0; i < size; ++i)
// 	{
// 		for (int j = 0; j < size; ++j)
// 		{
//       // 维数组在内存中存储是线性连续的，可以计算出二维数组的偏移量，进而使用一级指针遍历二维数组。
// 			printf("%f   ", *(host_result + i * size + j));
// 		}
// 		printf("\n");
// 	}
// }

// 使用指向一维数组的指针（一维数组的长度和二维数组的列数要一样）来遍历二维数组，
// 这样的好处就是，可以向使用二维数组名那样，通过下标来访问。
// void show_result(float (*host_result)[size], int size)
// {
//   printf("The result of Matrix multiplication is: \n");
// 	for (int i = 0; i < size; ++i)
// 	{
// 		for (int j = 0; j < size; ++j)
// 		{
//       // 维数组在内存中存储是线性连续的，可以计算出二维数组的偏移量，进而使用一级指针遍历二维数组。
// 			printf("%f   ", host_result[i][j]);
// 		}
// 		printf("\n");
// 	}
// }


int main(int argc, char *argv[])
{
  

  // Define host and device arrays.
  float host_a[size][size], host_b[size][size], host_result[size][size];
  float *device_a, *device_b, *device_result;

  // fill host matrix.
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      host_a[i][j] = i;
      host_b[i][j] = j;
    }
  }

  // Malloc memory for device.
  cudaMalloc((void**)&device_a, size * size * sizeof(int));
  cudaMalloc((void**)&device_b, size * size * sizeof(int));
  cudaMalloc((void**)&device_result, size * size * sizeof(int));

  // Copy data from host to device.
  cudaMemcpy(device_a, host_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

  // 使用 dim3 结构体定义 Grid 中的块和块中的线程形状，提前计算好的。
  // 多维线程的使用。
  dim3 dimGrid(size / TILE_SIZE, size / TILE_SIZE, 1);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

  // kernel call with nonshared memory.
  // gpu_matrix_multiplcation_nonshared <<< dimGrid, dimBlock >>> (device_a, device_b, device_result, size);
  
  // kernel call with shared memory.
  gpu_matrix_multiplcation_shared <<< dimGrid, dimBlock >>> (device_a, device_b, device_result, size);

  // Copy data from device to host.
  cudaMemcpy(host_result, device_result, size * size * sizeof(int), cudaMemcpyDeviceToHost);

  // Print result in concole.
  printf("The result of Matrix multiplication is: \n");
  for (int i = 0; i < size; ++i)
  {
  	for (int j = 0; j < size; ++j)
  	{
      // 维数组在内存中存储是线性连续的，可以计算出二维数组的偏移量，进而使用一级指针遍历二维数组。
  		printf("%f \t", host_result[i][j]);
  	}
  	printf("\n");
  }

  // Free up dynaming memory.
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_result);

  return 0;
}