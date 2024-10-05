/**
 * @file 00_memory_alignment.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-10-04
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <cstddef>
#include <iostream>

#define FIELD_OFFSET(TYPE, MEMBER) (size_t)(&(((TYPE *)0)->MEMBER))

/**
 * @brief 内存对齐规则
 * *1.结构体的数据成员, 第一个成员的偏移量为0, 后面每个数据成员存储的起始位置要从自己大小的整数倍开始;
 * *2.子结构体中的第一个成员偏移量应当是子结构体中最大成员的整数倍;
 * *3.结构体总大小必须是其内部最大成员的整数倍;
 * 
 * ?查看结构体偏移量
 * ?使用神器 0(0可以转换为任意类型的NULL指针)
#define FIELD_OFFSET(TYPE, MEMBER) (int)(&(((TYPE*)0)->MEMBER))
 * 
 */
struct Info
{
    char   username[10]; // 0-10 byte
    double userdata;     // 16-24 byte
};

struct Frame
{
    unsigned int   id;     // 0-1 byte
    unsigned int   width;  // 4-8 byte
    long long      height; // 8-16 byte
    unsigned char *data;   // x86 16-20 byte; x64 16-24
};

struct FrameInfo
{
    unsigned int   id;     // 0-1 byte
    unsigned int   width;  // 4-8 byte
    long long      height; // 8-16 byte
    unsigned char *data;   // x86 16-20 byte; x64 16-24
    Info           info;   // 24-(24-34 byte; 40-48byte)
};

#pragma pack(push)
#pragma pack(1) // 设置内存对齐为 1 byte

struct InfoAlignment
{
    char   username[10]; // 0-10 byte
    double userdata;     // 10-18 byte
};

#pragma pack(pop)

// ====================================
int main(int argc, const char **argv)
{
    std::cout << "The bytes size of Frame Struct ---> " << sizeof(Frame) << '\n';
    std::cout << "The bytes size of Info Struct ---> " << sizeof(Info) << '\n';
    std::cout << "The bytes size of Info Struct ---> " << sizeof(FrameInfo) << '\n';

    // =========================
    auto offset_width_field = FIELD_OFFSET(Frame, width);
    std::cout << "The width field offset in Frame ---> " << offset_width_field << '\n';

    auto offset_data_field = FIELD_OFFSET(Frame, data);
    std::cout << "The data field offset in Frame ---> " << offset_data_field << '\n';

    auto offset_info_field = FIELD_OFFSET(FrameInfo, info);
    std::cout << "The info field offset in FrameInfo ---> " << offset_info_field << '\n';

    std::cout << "The bytes size of InfoAlignment Struct ---> " << sizeof(InfoAlignment) << '\n';

    return 0;
}
