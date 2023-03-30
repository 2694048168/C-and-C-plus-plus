#include <cstdio>
// #include <iostream>

__global__ void hello_cuda()
{
    /* 核函数(kernel function)中不支持 C++ 的 iostream . */
    // std::cout << "Hello CUDA from GPU." << std::endl;
    printf("Hello CUDA from GPU.\n");
}

int main(int argc, char const *argv[])
{

    hello_cuda<<<1, 1>>>();
    
    /* CUDA 的运行时 API 函数 cudaDeviceSynchronize 的作用
    是同步主机(host)与设备(device), 所以能够促使缓冲区刷新,
    否则打印不出字符串. */
    cudaDeviceSynchronize();
    
    return 0;
}
