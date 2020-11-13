#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024  // 向量中元素的个数。

// 每一个block中线程数量，这样将内核配置参数定义为常量，便于程序的移植和修改。
#define threadsPerBlock 512

/**向量点乘，向量内积运算
 * 向量内积运算，就是将多个对应元素的乘法结果累加起来。
 *
 * CUDA 编程中重要概念：归约运算。
 * 这种原始输入是两个数组，而输出为一个单一的数值的运算，CUDA 编程称之为归约运算。 
 */
// Define kernel function to compute the vector dot preduct.
__global__ void gpu_vector_dot(float *device_a, float *device_b, float *device_c)
{
  // Define shared memory.
  // 共享内存中的元素等于每个块的线程数，每一个块都有自己单独的共享内存地址。
  __shared__ float partial_sum[threadsPerBlock];

  // Computer the current unique thread ID.
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // 作为每次计算部分元素和的索引。
  unsigned int index = threadIdx.x;

  float sum = 0;
  while (tid < N)
  {
    // 计算向量对应元素的乘积结果
    sum += device_a[tid] * device_b[tid];

    // 由于配置的并行启动的线程总数有限，不可能达到每次并行计算需要的总数 N。
    // 所以，每个线程必须执行多个操作，由已启动的线程总数作为偏移量进行分隔。
    // 偏移量计算 tid += blockDim.x * gridDim.x;
    // 前者代表了本次启动的块的数量，后者代表了每个块里面的线程数量，
    // 这样向后偏移以得到下一个任务的索引。
    tid += blockDim.x * gridDim.x;
  }

  // Set the partial sum in shared memory.
  partial_sum[index] = sum;

  // synchronize threads in the block.
  // 在进行归约运算之前，即对共享内存中的数据进行读取之前，必须保证每个线程都完成对共享内存的写入操作。
  // __syncthreads(); 同步函数可以做到。
  __syncthreads();

  // Calculate partial sum for a current block using data in shared memory.
  // Why should we do this as follow ?
  // 将每一个块中的部分和结果，并行化执行累加，并将2个数累加结果覆盖写入第一个数的位置地址，
  // 重复该累加方式，最终得到整个块的部分和结果，并将块中的部分和结果以块的ID 为索引写入全局内存中。
  unsigned int i = blockDim.x / 2;
  while (i != 0)
  {
    if (index < i)
    {
      // 并行化执行累加，并将2个数累加结果覆盖写入第一个数的位置地址，
      partial_sum[index] += partial_sum[index + i];
    }
    // synchronize threads 同步操作。
    __syncthreads();

    // 怎么完成并行化操作的索引？
    i /= 2;
  }
  // Store result of partial sum for a block in global memory.
  if (index == 0)
  {
    // 整个块的部分和结果。
    device_c[blockIdx.x] = partial_sum[0];
  }
}


int main(int argc, char *argv[])
{
  // Define the host and device pointers.
  float *host_a, *host_b, host_c, *partial_sum;
  float *device_a, *device_b, *device_partial_sum;

  // Calculate number of blocks and number of threads.
  unsigned int block_calc = (N + threadsPerBlock - 1) / threadsPerBlock;
  unsigned int blocks_per_grid = (32 < block_calc ? 32 : block_calc);

  // Allocate memory on the CPU host.
  host_a = (float*)malloc(N * sizeof(float));
  host_b = (float*)malloc(N * sizeof(float));
  partial_sum = (float*)malloc(blocks_per_grid * sizeof(float));

  // Allocate memory on the GPU device.
  cudaMalloc((void**)&device_a, N * sizeof(float));
  cudaMalloc((void**)&device_b, N * sizeof(float));
  cudaMalloc((void**)&device_partial_sum, blocks_per_grid * sizeof(float));

  // Fill in the host memory with data.
  for (unsigned int i = 0; i < N; ++i)
  {
    // 等差数列的求和，便于后面的CPU计算测试。
    host_a[i] = i;
    host_b[i] = 2;
  }

  // Copy the arrays data on host to the device.
  cudaMemcpy(device_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);

  // kernel call.
  gpu_vector_dot <<< blocks_per_grid, threadsPerBlock >>> (device_a, device_b, device_partial_sum);

  // Copy the arrays data on device back to the host.
  cudaMemcpy(partial_sum, device_partial_sum, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);

  // Calculate final vector dot preduct.
  host_c = 0;
  for (unsigned int i = 0; i < blocks_per_grid; ++i)
  {
    // 将每一个块计算的部分和，通过块的ID 索引的值，计算最终的结果。
    host_c += partial_sum[i];
  }

  // 检查向量内积的结果是否正确。
  printf("The computed dot product is: %f\n", host_c);
  // 等差数列的求和。
  #define cpu_sum(x) (x * (x+1))
  /**GPU 和 CPU 的浮点运算结果不能通过 == 直接进行对比。
   * 这是因为GPU 并行计算的本质，达到的浮点数结果几乎总是和 CPU 的结果在最后的尾数上有轻微的差异。
   * 建议对两个结果进行作差，得到其绝对值，当其值足够小，就认为结果一致。
   */
	if (host_c == cpu_sum((float)(N - 1)))
	{
		printf("The dot product computed by GPU is correct.\n");
	}
	else
	{
		printf("Error in dot product computation.");
	}

  // Free dynamic allocation memory on host and device.
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_partial_sum);
	free(host_a);
	free(host_b);
	free(partial_sum);

  return 0;
}