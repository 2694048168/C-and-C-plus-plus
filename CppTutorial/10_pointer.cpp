/**
 * @file 10_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程之指针和智能指针
 * @version 0.1
 * @date 2024-03-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <iostream>

// ====================================
int main(int argc, const char **argv)
{
    void *ptr   = malloc(sizeof(int));
    *(int *)ptr = 42;
    std::cout << "变量 ptr 的地址: " << &ptr << std::endl;
    std::cout << "变量 ptr 的地址: " << static_cast<void *>(&ptr) << std::endl;
    std::cout << "变量 ptr 所存储的数值: " << *(int *)ptr << std::endl;
    std::cout << "变量 ptr 的地址 " << static_cast<const void *>(ptr) << std::endl;

    std::cout << "\n指针变量占据字节大小: " << sizeof(ptr) << std::endl;
    std::cout << "指针变量占据字节大小: " << sizeof(void *) << std::endl;
    std::cout << "指针变量占据字节大小: " << sizeof(int *) << std::endl;

    float *ptr_ = new float;
    std::cout << "\n指针变量占据字节大小: " << sizeof(ptr_) << std::endl;

    // 申请一块内存地址, 用于存储从相机(黑白16K线扫相机)采集的图像
    const unsigned int imageWidth   = 16384;
    const unsigned int imageHeight  = 5000;
    unsigned char     *pImageBuffer = (unsigned char *)malloc(imageWidth * imageHeight);
    std::cout << "\npImageBuffer 指针变量的地址: " << &pImageBuffer << std::endl;
    std::cout << "pImageBuffer 指向的内存地址: " << pImageBuffer << std::endl;
    std::cout << "pImageBuffer 指向的内存地址: " << (void *)pImageBuffer << std::endl;

    // NOTE: 记得当不需要此块内存时, 需要释放申请的内存
    if (ptr != nullptr)
    {
        free(ptr);
        ptr = nullptr;
    }
    if (ptr_ != nullptr)
    {
        delete ptr_;
        ptr_ = nullptr;
    }
    if (pImageBuffer != nullptr)
    {
        free(pImageBuffer);
        pImageBuffer = nullptr;
    }

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\10_pointer.cpp -std=c++23
// g++ .\10_pointer.cpp -std=c++23
