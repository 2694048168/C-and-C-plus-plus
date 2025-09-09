/**
 * @file 40_clamp.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Modern C++17 std::clamp function
 * @version 0.1
 * @date 2025-09-09
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm> // std::clamp
#include <cstdint>
#include <iomanip>
#include <iostream>

/**
 * @brief 图像处理的算法，其中一个需求是将其他颜色域的色彩转换为 RGB 颜色分量，
 * 因为其他颜色域与 RGB 颜色域并不完全一致，总有一些色彩转换后出现非法的值，需要做校正.
 * ? r = std::clamp(r, 0, 255);
 * std::clamp() 的作用是提供一种标准且安全的方式将值限制在指定范围内的方法，
 * 避免手动实现时可能出现的边界条件错误或冗余代码.
 * 它通过简洁的语法强制将输入值“夹紧”（clamp）在 [min, max] 区间内(闭区间)，
 * 确保结果始终处于可控范围内.
 * 
 * g++ 40_clamp.cpp -std=c++17
 * 
 */

// -------------------------------------
int main(int argc, const char *argv[])
{
    std::cout << "[raw] "
                 "["
              << INT8_MIN << ',' << INT8_MAX
              << "] "
                 "[0,"
              << UINT8_MAX << "]\n";

    for (const int v : {-129, -128, -1, 0, 42, 127, 128, 255, 256})
        std::cout << std::setw(4) << v << std::setw(11) << std::clamp(v, INT8_MIN, INT8_MAX) << std::setw(8)
                  << std::clamp(v, 0, UINT8_MAX) << '\n';

    return 0;
}
