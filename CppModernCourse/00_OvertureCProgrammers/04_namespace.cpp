/**
 * @file 04_namespace.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief 命名空间与结构体、联合体和枚举的隐式类型定义 
 * C++ 把类型标记当做隐式的 typedef 名称, C语言需要使用 typedef 为创建的类型指定名称,
 * 命名空间(namespace), 为标识符创建不同的作用域, 有助于保存用户类型和函数的整洁,
 * 使用 namespace 是 C++ 的惯例, 是一种零开销的抽象, 在生成汇编代码时被编译器清除,
 * 可以使用 using 关键字进行 namespace 的导入, 个人不建议这样做, 失去了原本的作用,
 * 
 */
// ========== C-style ==========
typedef struct Jabberwocks
{
    void *_wood;
    int   is_galumphing;
} Jabberwock;

// ========== C++ style ==========
enum JabberwocksEnum
{
    WOOD = 0,
    NUM_ENUM
};

namespace Creature {
void PrintInfo(const char *str)
{
    std::cout << "Creature::PrintInfo:\t";
    std::cout << str << '\n';
}
} // namespace Creature

namespace Func {
void PrintInfo(const std::string &str)
{
    std::cout << "Func::PrintInfo:\t";
    std::cout << str << '\n';
}
} // namespace Func

// -----------------------------------
int main(int argc, const char **argv)
{
    Creature::PrintInfo("const char *str");
    Func::PrintInfo("std::string &str");

    return 0;
}
