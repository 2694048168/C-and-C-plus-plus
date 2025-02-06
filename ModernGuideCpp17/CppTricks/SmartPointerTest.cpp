/*****************************************************************//**
 * \file   SmartPointerTest.cpp
 * \brief  智能指针和裸指针的性能开销差异测试(汇编指令), 编译器优化
 *
 * \author WeiLi (Ithaca)
 * \date   February 2025
 *********************************************************************/

 // SmartPointerTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
 // 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
 // 调试程序: F5 或调试 >“开始调试”菜单
 // F9 添加断点 ---> F5 进入调试 ---> Ctrl + K + G 跳转到反汇编代码 

#include <iostream>
#include <memory>

class XData
{
public:
    void Test()
    {
        m_index += rand();
    }

    void Print()
    {
        std::cout << m_index;

    }

private:
    int m_index{ 0 };
    char m_buffer[1024]{ 0 };
};

//---------------------------------------------
int main(int /*argc*/, const char** /*argv*/)
{
    std::cout << "Hello World!\n";

    // 创建性能消耗
    auto pXdata = new XData(); // new 函数调用开销
    auto spXdata = std::make_unique<XData>(); // new 函数调用开销 + 智能指针对象创建开销
    // 实际编译器优化后, 两者都是直接调用 new函数, 没有创建性能差异

    // 空间占用
    std::cout << "Space ordinary-pointer: " << sizeof(pXdata) << '\n';
    std::cout << "Space smart-pointer: " << sizeof(spXdata) << '\n';

    // 调用性能消耗
    pXdata->Test(); // 直接调用函数地址
    spXdata->Test(); // 智能指针对象调用重载的 '->' 然后调用函数地址;
    // 实际编译器优化后, 两者都是直接调用函数地址, 没有调用性能差异

    return 0;
}
