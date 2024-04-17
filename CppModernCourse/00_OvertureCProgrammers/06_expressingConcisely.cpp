/**
 * @file 06_expressingConcisely.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <iostream>
#include <vector>

/**
 * @brief 简洁地表达想法和重用代码 Expressing Ideas Concisely and Reusing Code,
 * 精心设计的C++代码很优雅, 很紧凑, 感受从 ANSI-C 到 Modern C++ 的演进,
 * 遍历有 n 个元素的数组∨
 * 
 */

struct Position
{
    float x;
    float y;
    float z;
};

void navigate_to(const Position &p);
// Position *get_position();            // method 1
void get_position(Position *p); // method 2

/** 在第一种方法中, 调用者负责清理返回值, 这可能产生一个动态内存分配;
 * 在第二种方法中, 调用者负责分配一个 Position 并把它传入 get position, 更符合C语言的习惯;
 * 本来只想得到一个位置对象, 却不得不担心是调用者还是被调用者负责分配和删除内存.
 * C++ 可以通过直接从函数中返回用户自定义类型来简洁地完成这一切,
 */

Position get_position()
{
    auto position_obj = new Position;
    return *position_obj;
}

/**
 * 因为 get position 返回一个值, 编译器可以忽略这个复制操作,
 * 所以就好像直接构造了一个自动的 Position 变量, 没有运行时开销,
 * 从功能上讲，这个情况与 method 2 中的C风格指针传递非常相似.
 * 
 */
void navigate()
{
    auto pos = get_position();
    // p is now available for use
    std::cout << pos.x << " " << pos.y << " " << pos.z << '\n';
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::vector<int> vec_values = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    // ======== ANSI-C ========
    std::cout << "\n======== ANSI-C ========\n";
    size_t idx;
    for (idx = 0; idx < vec_values.size(); idx++)
    {
        std::cout << vec_values[idx] << " ";
    }

    // ======== C99 ========
    std::cout << "\n======== C99 ========\n";
    for (size_t idx = 0; idx < vec_values.size(); idx++)
    {
        std::cout << vec_values[idx] << " ";
    }

    // ======== C++17 ========
    std::cout << "\n======== C++17 ========\n";
    for (const auto &elem : vec_values)
    {
        std::cout << elem << " ";
    }

    
    std::cout << "\n======== navigate ========\n";
    navigate();

    return 0;
}
