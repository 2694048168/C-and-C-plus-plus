module math; // 注意: 这里没有 export, 表示正在为 "math" 模块提供实现

// 包含实现所需的头文件，这些头文件不会污染到 import math 的地方
#include <numeric>

int add(int a, int b)
{
    return a + b;
}

// 函数的具体实现
int sub(int a, int b)
{
    // 假设这里有一个复杂的实现
    return a - b;
}

// 这个函数没有被 export，它就是模块的私有部分
int internal_helper()
{
    return 42;
}
