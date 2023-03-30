#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

__global__ void hello_cuda()
{
    /* kernel function built-in variable: warpSize(线程束)
    gridDim.x | blockDim.x | blockIdx | threadIdx
    gridDim.y | blockDim.y | blockIdy | threadIdy
    gridDim.z | blockDim.z | blockIdz | threadIdz
    ---------------------------------------------- */
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id_x = threadIdx.x;
    const unsigned int thread_id_y = threadIdx.y;
    printf("The kernel function calling warpSize %d\n", warpSize);
    printf("Hello CUDA from block %d and thread (%d, %d) on GPU.\n",
           block_id, thread_id_x, thread_id_y);
}

int main(int argc, char const *argv[])
{
    /* CUDA 源代码中获取 GPU 设备的属性 */
    int device_count = 0; // 可用设备数量
    int device;           // 设备名称编号
    int driver_version;   // CUDA 驱动版本号
    int runtime_version;  // 运行时引擎版本

    /* This function returns count of number of CUDA enable devices
     and 0 if there are no CUDA capable devices.
     ------------------------------------------- */
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cout << "There are no available devices that support CUDA.\n";
    }
    else
    {
        std::cout << "Detected " << device_count << " CUDA Capable devices.\n";
    }

    /* 通过查询 cudaDeviceProp 结构体来找到每个设备的相关信息，该结构体返回所有设备的属性。
    如果有多个可用设备，可使用 for 循环遍历所有设备属性。
    get count of the avaiable CUDA hardware device in system.
    --------------------------------------------------------- */
    /**通用设备信息
     * cudaDeviceProp 结构体提供可以用来识别设备以及确认使用的版本信息的属性，
     * name属性以字符串形式返回设备名称。
     * cudaDriverGetVersion 获取设备使用的 CUDA Driver。
     * cudaRuntimeGetVersion 获取设备运行时引擎的版本。
     * clockRate属性获取 GPU 的时钟速率。
     * multiProcessorCount属性用于判断设备上流多处理器的个数。
     --------------------------------------------------- */
    cudaGetDevice(&device);
    std::cout << "ID of device: " << device << std::endl;

    cudaDeviceProp device_property;
    cudaGetDeviceProperties(&device_property, device);
    std::cout << "The type of hardware GPU is: " 
            << device_property.name << std::endl;

    cudaDriverGetVersion(&driver_version);
    std::cout << "CUDA Driver Version is: CUDA " 
            << driver_version / 1000 << "." 
            << (driver_version % 100) / 10 << std::endl;

    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA Runtime Driver Version is: CUDA " 
            << driver_version / 1000 << "." 
            << (driver_version % 100) / 10 << std::endl;

    std::cout << "Total amount of global memory: " 
            << (float)device_property.totalGlobalMem / 1048576.0f 
            << " MBytes" << std::endl;

    // 具有最多流处理器的设备,如果有多个设备
    printf(" (%2d) mutilprocessors\n", device_property.multiProcessorCount);

    // GPU 时钟速率，以 KHz 为单位进行返回
    std::cout << "GPU max clock rate: " 
            << device_property.clockRate * 1e-6f << " GHz" << std::endl;

    // 显存频率
    std::cout << "Memory clock rate: " 
            << device_property.memoryClockRate * 1e-3f << " MHz" << std::endl;

    std::cout << "Memory Bus Width: " 
            << device_property.memoryBusWidth << "-bit" << std::endl;

    if (device_property.l2CacheSize)
    {
        std::cout << "L2 Cache size: " 
                << device_property.l2CacheSize << " bytes" << std::endl;
    }

    std::cout << "Toal amount of constant memory: " 
            << device_property.totalConstMem << " bytes" << std::endl;

    std::cout << "Toal amount of shared memory per block: " << device_property.sharedMemPerBlock << " bytes" << std::endl;

    // 每一个块可用寄存器总数
    std::cout << "Toal amount of registers available per block: " 
            << device_property.regsPerBlock << std::endl;

    // 网格grid 块block 线程thread 可以时多维的，每一个维度中可用并行启动多少个线程和块
    // 这对于内核参数的配置十分重要
    std::cout << "Maximum number of threads per block: " 
            << device_property.maxThreadsPerBlock << std::endl;

    std::cout << "Max dimension size of a thread block (x, y, z): "
              << "(" << device_property.maxThreadsDim[0] << "," 
              << device_property.maxThreadsDim[1] << "," 
              << device_property.maxThreadsDim[2] << ")" << std::endl;

    std::cout << "Max dimension size of a grid size (x, y, z): "
              << "(" << device_property.maxGridSize[0] << "," 
              << device_property.maxGridSize[1] << "," 
              << device_property.maxGridSize[2] << ")" << std::endl;

    // 查看设备是否支持双精度浮点操作，并为应用程序设置该设备
    memset(&device_property, 0, sizeof(cudaDeviceProp));
    // if major > 1 and minor > 3, then the device supports double precision.
    device_property.major = 1;
    device_property.minor = 3;

    // 选择特定属性的设备
    cudaChooseDevice(&device, &device_property);

    std::cout << "ID of device which supports double precision is: " 
            << device << std::endl;
    // 设置设备为应用程序所用设备
    cudaSetDevice(device);

    /* 推广到多维网格, 多维的网格和线程块本质上还是一维的,
    就像多维数组本质上也是一维数组一样. 网格与线程块大小有限制.
    核心思想还是 CUDA 的计算核心的布局和如何索引.
    --------------------------------------------- */
    const dim3 block_size(2, 4);
    hello_cuda<<<1, block_size>>>();

    cudaDeviceSynchronize();

    return 0;
}
