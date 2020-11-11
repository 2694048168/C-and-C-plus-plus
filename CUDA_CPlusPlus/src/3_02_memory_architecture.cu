/**Nvidia GPU 存储器架构
 * 在 GPU 上代码的执行被划分为流多处理器SM，Streaming Multiprocessor；块block；线程thread。
 * 每一个存储器空间都有其特定的特征和用途，不同的访问速度和生命周期范围，详情查看具体 GPU 存储器架构图。
 * 全局内存global memory；常量内存constant memory；纹理内存texture memory；
 * 共享内存shared memory；寄存器堆registers；本地内存local memory。
 * 高速缓存存储器，L1 L2 缓存，提高访问速度。
 * CUDA 存储器的快慢有两方面：延迟低；带宽大；一般都是指延迟低。
 */

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

// Define kernel function with global memory.
__global__ void gpu_global_memory(int *device_a)
{
  device_a[threadIdx.x] = threadIdx.x;
}

/**本地内存和寄存器堆对每一个线程都是唯一的。
 * 寄存器溢出，有两种情况：
 * 1、寄存器不够了；
 * 2、某些情况下根本就不能放在寄存器中。
 */
// Define kernel function with local memory.
__global__ void gpu_local_memory(int device_in)
{
  // 在 kernel function 内部声明的局部变量就是寄存器或者本地存储器中。
	int t_local;    
	t_local = device_in * threadIdx.x;     
	printf("Value of Local variable in current thread is: %d \n", t_local);
}

// Define kernel function with shared memory.
__global__ void gpu_shared_memory(float *device_a)
{
	// Defining local variables which are private to each thread.
	int i, index = threadIdx.x;
  float average, sum = 0.0f;
  
  //Define shared memory.
  // __shared__ 限定修饰的变量，存储在共享内存中。
	__shared__ float sh_arr[10];

	
	sh_arr[index] = device_a[index];

	__syncthreads();    // This ensures all the writes to shared memory have completed

  // finish MA 操作，就是计算数组中当前元素之前所有元素的平均值。
	for (i = 0; i<= index; ++i) 
	{ 
		sum += sh_arr[i]; 
	}
	average = sum / (index + 1.0f);

	device_a[index] = average; 

	sh_arr[index] = average;
}

int main(int argc, char *argv[])
{
  int host_a[N];
  int *device_a;

  // Malloc global memory on device.
  // cudaMalloc 分配的存储器都是全局内存。
  cudaMalloc((void**)&device_a, sizeof(int) * N);
  // Copy host memory to device global memory.
  cudaMemcpy((void*)&device_a, (void*)host_a, sizeof(int) * N, cudaMemcpyHostToDevice);
  // Kernel call.
  gpu_global_memory <<<1, N>>> (device_a);
  // Copy device global memory to host memory.
  cudaMemcpy((void*)&host_a, (void*)device_a, sizeof(int) * N, cudaMemcpyDeviceToHost);
  //Testing the global memory.
  printf("Array in global memory is: \n");
  for (unsigned int i = 0; i < N; ++i)
  {
    printf("At index: %d ---> %d\n", i, host_a[i]);
  }

  // Testing the local memory.
  printf("Use of Local Memory on GPU:\n");
	gpu_local_memory << <1, N >> > (N);  
  cudaDeviceSynchronize();
  
  // Testing the shared memory.
	float h_a[10];   
	float *d_a;       
	
  for (int i = 0; i < 10; ++i) 
  {
		h_a[i] = i;
	}
	// allocate global memory on the device
	cudaMalloc((void **)&d_a, sizeof(float) * 10);
	// now copy data from host memory  to device memory 
	cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
	// Call kernel function.
	gpu_shared_memory << <1, 10 >> >(d_a);
	// copy the modified array back to the host memory
	cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	printf("Use of Shared Memory on GPU:  \n");
	//Printing result on console
  for (int i = 0; i < 10; ++i) 
  {
		printf("The running average after %d element is %f \n", i, h_a[i]);
	}

  return 0;
}