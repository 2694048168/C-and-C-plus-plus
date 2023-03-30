#include <cstdio>
#include <iostream>

__global__ void hello_cuda()
{
    /* kernel function built-in variable:
    gridDim.x | blockDim.x | blockIdx | threadIdx 
    ---------------------------------------------- */
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    printf("The kernel function calling GridSize %d and BlockSize %d\n",
            gridDim.x, blockDim.x);
    printf("Hello CUDA from block %d and thread %d on GPU.\n", 
            block_id, thread_id);
}

int main(int argc, char const *argv[])
{

    /* Multiple thread by CUDA computing core.
    gridSize X blockSize = 2 x 4 = 8 threads.
    <<<grid_size, block_size>>> 
    ------------------------------------------ */
    hello_cuda<<<2, 4>>>();
    // std::cout << "The kernel function calling GridSize: " << gridDim.x 
    //         << "and calling BlockSize: " << blockDim.x << std::endl;
    
    cudaDeviceSynchronize();
    
    return 0;
}
