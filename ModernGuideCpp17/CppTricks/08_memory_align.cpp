/**
 * @file 08_memory_align.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <iostream>

/**
 * @brief C++ 编程中, 内存对齐(memory align)是一个既常见又复杂的概念.
 * 对齐不当不仅会影响程序性能, 甚至会导致程序崩溃(crash)或未定义行为(undefined behavior)
 * since C++11 引入的 alignas 和 alignof 关键字, 则为开发者提供了对内存对齐的精确控制.
 * 
 * *什么是内存对齐?
 * 内存对齐是指数据在内存中按照特定的字节边界存储;
 * 一般情况下, 处理器从内存读取数据时会更高效地读取对齐的数据;
 * 如果数据未对齐, 处理器可能需要进行额外的内存访问, 导致性能下降;
 * 对于某些平台(ARM or PLC or FPGA), 不对齐的内存访问甚至可能引发未定义行为.
 * 
 * ?通常, C++ 编译器(compiler)会根据目标平台自动对变量(struct)进行内存对齐,
 * 确保类型的内存地址是适当的对齐边界. 但有时, 开发者需要对内存对齐进行精确控制,
 * 比如优化性能, 与硬件设备交互等场景, 这就是 alignas 和 alignof 的用武之地.
 * 
 * ?compiler 内存对齐规则
 * 每个特定平台上的编译器都有自己的默认"对齐系数"(也叫对齐模数),
 * GCC 中默认 #pragma pack(4),
 * 可以通过预编译命令 #pragma pack(n), n = 1,2,4,8,16 来改变这一系数.
 * 
 * ?常见应用场景
 * *1. 性能优化: 在高性能计算场景中, 合理的内存对齐可以显著提升程序性能.
 * 例如, 使用 SIMD(单指令多数据)指令集的处理器通常要求数据以 16 字节或更高对齐.
 * *2. 硬件接口: 当与硬件设备交互时，硬件可能要求数据按照特定的字节边界对齐.
 * 这时 alignas 可以帮助开发者满足硬件对齐要求, 避免读取或写入错误.
 * 
 */

// 使用 alignas 指定结构体成员的对齐方式,以满足更复杂的内存布局需求
struct MyStruct
{
    char a;
    alignas(16) double b; // double b 对齐到 16 字节
    int c;
};

// alignas 还可以与类型结合使用, 指定变量的对齐方式与另一种类型相同
struct alignas(double) AlignedStruct
{
    int    x;
    double y;
};

// 在模板编程中的应用
// alignof 在模板编程中尤为实用, 特别是希望根据类型的对齐方式动态调整数据结构或算法时.
template<typename T>
void print_alignment()
{
    std::cout << alignof(T) << " 字节\n";
}

struct CustomType
{
    std::string  name;
    unsigned int width;
    unsigned     height;
    double       precision;
    float        depth;
};

// alignas 和 alignof 的配合使用
// 在某些场景中, alignas 和 alignof 可以结合使用,
// 可以使用 alignof 来确定系统中某个类型的对齐要求, 并通过 alignas 将其他数据对齐到相同的标准.
// ?通过模板动态确定类型 T 的对齐要求, 并使用 alignas 为存储空间 storage 提供正确的对齐方式.
template<typename T>
struct AlignedStorage
{
    alignas(alignof(T)) char storage[sizeof(T)];
};

struct Demo
{
    alignas(128) char c[128];
    char c1;
    int  n;
    char c2;
    char c3;
    alignas(256) char c4[128];
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // ===== alignas: 显式指定对齐要求
    /** alignas 关键字允许显式指定变量、对象、结构体或类的对齐要求,
      通过 alignas, 可以将数据对齐到指定的字节边界:
      alignas(alignment) type variable;
      alignment: 一个整数常量, 指定对齐字节数, 
    * !必须是2的幂, 如 2, 4, 8, 16 等;
    * !自定义对齐方式不得小于默认的最小对齐方式;
      type: 变量或类型的声明;
    */
    alignas(16) int   data[4]; // data 数组的对齐要求是 16 字节?
    alignas(16) float value;

    std::cout << "data 对齐字节数: " << alignof(decltype(data)) << '\n';
    std::cout << "data 对齐字节数: " << alignof(data) << '\n';
    std::cout << "value 对齐字节数: " << alignof(decltype(value)) << '\n';
    std::cout << "value 对齐字节数: " << alignof(value) << '\n';

    std::cout << "MyStruct 的对齐字节数: " << alignof(MyStruct) << '\n';
    std::cout << "AlignedStruct 的对齐字节数: " << alignof(AlignedStruct) << '\n';

    /* alignof: 查询类型的对齐要求
     alignof 关键字用于查询某种类型或对象的对齐方式, 
     返回一个 std::size_t 值, 表示该类型需要的对齐字节数:
     alignof(type)
     type: 要查询对齐方式的类型. 
     */
    std::cout << "int 的对齐字节数: " << alignof(int) << '\n';
    std::cout << "double 的对齐字节数: " << alignof(double) << '\n';
    std::cout << "char 的对齐字节数: " << alignof(char) << '\n';

    // 在模板编程中的应用
    print_alignment<int>();
    print_alignment<double>();
    print_alignment<CustomType>();

    // 特殊应用
    AlignedStorage<int> alignedInt;
    std::cout << "AlignedStorage<int> 对齐字节数: " << alignof(decltype(alignedInt)) << std::endl;

    // 确定并计算结构体成员的内存地址
    std::cout << alignof(Demo) << std::endl;      //256
    std::cout << offsetof(Demo, c) << std::endl;  //0
    std::cout << offsetof(Demo, c1) << std::endl; //128
    std::cout << offsetof(Demo, n) << std::endl;  //132
    std::cout << offsetof(Demo, c2) << std::endl; //136
    std::cout << offsetof(Demo, c3) << std::endl; //137
    std::cout << offsetof(Demo, c4) << std::endl; //256

    return 0;
}
