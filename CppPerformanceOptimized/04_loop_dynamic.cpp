/**
 * @file 04_loop_dynamic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <list>
#include <string>
#include <vector>

// 但是随着 List 变大，这个开销将会降低程序性能
// 如果它有 1000 个元素，那么内存管理器会被调用 1000 次
// 在函数最后，形参 v 会超出它的作用域，其中的 1000 个元素也会被逐一返回给不会再被使用的链表
// *通过复制构造函数创建它的实例可能会调用内存管理器来复制它内部的数据，
// *而传递指向类实例的引用可以改善程序性能
int Sum(std::list<int> v)
{
    int sum = 0;
    for (auto it : v)
    {
        sum += it;
    }
    return sum;
}

// =====================================
int main(int argc, const char *argv[])
{
    std::vector<std::string> nameVec;

    for (auto &filename : nameVec)
    {
        std::string config;
        // ReadFileXML(filename, config);
        // ProcessXML(config);
    }

    // ------ 在循环外创建动态变量 --------
    std::string config;
    for (auto &filename : nameVec)
    {
        config.clear();
        // ReadFileXML(filename, config);
        // ProcessXML(config);
    }

    // 延长动态分配内存的变量的生命周期可以显著提升性能
    // 这种技巧不仅适用于 std::string, 也适用于 std::vector 和其他任何拥有动态大小骨架的数据结构
    // 由于每个线程都会请求内存管理器来为记录信息分配内存,这种竞争内存管理器的现象导致性能陡然下降

    return 0;
}
