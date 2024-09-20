/**
 * @file 05_union_application.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief 联合体应用: 验证当前主机的大小端（字节序）
 * *====大小端(Endianness)指的是在多字节数据类型(如整数)在内存中存储时的字节顺序;
 * 1. 大端字节序(Big Endian)是指高位字节存储在低地址;
 * 2. 小端字节序(Little Endian)是指低位字节存储在低地址;
 * 
 * 举个例子有一个 4 字节的整数值 0x12345678 在内存中存储的情况如下:
 * 1. 大端字节序: 地址由低到高的顺序依次存储为 12 34 56 78
 * 2. 小端字节序: 地址由低到高的顺序依次存储为 78 56 34 12
 * ?不同的处理器架构在存储数据时可能采用不同的字节序.
 * 例如 x86 架构使用小端字节序, 而 PowerPC 架构使用大端字节序.
 * ?因此在处理跨平台数据交换时需要注意字节序的问题. 
 * 
 * ?内存布局：https://subingwen.cn/c/memory-layout/
 * ?预处理：https://subingwen.cn/c/preprocessing/
 * 
 */

#include <iostream>

union EndiannessBytesData
{
    unsigned int data;

    struct
    {
        unsigned char byte0;
        unsigned char byte1;
        unsigned char byte2;
        unsigned char byte3;
    } byte;
};

// ======================================
int main(int argc, const char **argv)
{
    union EndiannessBytesData num;
    num.data = 0x12345678;
    if (0x78 == num.byte.byte0)
    {
        std::cout << "Current Platform is Little endian\n";
    }
    else if (0x78 == num.byte.byte3)
    {
        std::cout << "Current Platform is Big endian\n";
    }

    return 0;
}
