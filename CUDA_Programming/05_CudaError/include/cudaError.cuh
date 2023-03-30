#ifndef _CUDA_ERROR_CUH_
#define _CUDA_ERROR_CUH_

// 定义宏函数
#define CHECK(call)\
do\
{\
    const cudaError_t error_code = call;\
    if (error_code != cudaSuccess)\
    {\
        std::cout << "CUDA Error: \n";\
        std::cout << "---- File: " << __FILE__ << "\n";\
        std::cout << "---- Line: " << __LINE__ << "\n";\
        std::cout << "---- Error code: " << error_code << "\n";\
        std::cout << "---- Describe: " << cudaGetErrorString(error_code) << "\n";\
        exit(1);\
    }\
} while (0);

#endif /* _CUDA_ERROR_CUH_ */