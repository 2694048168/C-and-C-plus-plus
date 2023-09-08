/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "test/test_a.hpp"
#include "test/test_b.hpp"
#include "test/a_custom.hpp"
#include "test/b_custom.hpp"
#include "utility/auto_ptr.hpp"
#include "utility/shared_ptr.hpp"
#include "utility/smart_ptr.hpp"
#include "utility/unique_ptr.hpp"
// #include "utility/weak_ptr.hpp"

using namespace WeiLi::utility;

#include <iostream>
#include <memory>
#include <string>

// global variable
class Test
{
public:
    Test() = default;

    ~Test()
    {
        std::cout << "this is destructor, and Test is delete!\n";
    }

    void name(const std::string &name)
    {
        m_name = name;
    }

    void name(const char *name)
    {
        m_name = name;
    }

    std::string name() const
    {
        return m_name;
    }

private:
    std::string m_name;
};

/**
 * @brief 智能指针的理解和使用
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    /* Step 1. RAII 技术，不需要显式释放资源，
    同时对象在其声明周期内资源是安全和有效的
    ------------------------------------- */
    {
        std::cout << "==================================\n";
        SmartPtr<Test> smart_ptr1(new Test());

        smart_ptr1->name("Wei Li");
        std::cout << "the name is: " << smart_ptr1->name() << std::endl;
        std::cout << "the name is: " << (*smart_ptr1).name() << std::endl;
    }

    /* Step 2. C++98 auto_ptr 智能指针, 已经被废弃了 
    ---------------------------------------------- */
    std::cout << "==================================\n";
    AutoPtr<Test> smart_ptr2(new Test());

    smart_ptr2->name("Wei Li");
    std::cout << "the name is: " << smart_ptr2->name() << std::endl;
    std::cout << "the name is: " << (*smart_ptr2).name() << std::endl;

    // 对象所拥有的所有权
    AutoPtr<Test> smart_ptr3(smart_ptr2);
    // std::cout << "the name is: " << (*smart_ptr2).name() << "\n"; /* running error */
    std::cout << "the name is: " << (*smart_ptr3).name() << std::endl;
    smart_ptr3->name("GuiYang");
    std::cout << "the name is: " << smart_ptr3->name() << std::endl;

    /* Step 3. C++11 unique_ptr 独占智能指针,
    对象所有权不能被转移, 但是可以移动 ^_^  
    --------------------------------------- */
    std::cout << "==================================\n";
    UniquePtr<Test> smart_ptr4(new Test());

    smart_ptr4->name("Wei Li");
    std::cout << "the name is: " << smart_ptr4->name() << std::endl;
    std::cout << "the name is: " << (*smart_ptr4).name() << std::endl;

    // 对象所拥有的所有权
    // UniquePtr<Test> smart_ptr5(smart_ptr4);
    // smart_ptr5 = smart_ptr4;

    UniquePtr<Test> smart_ptr5(std::move(smart_ptr4));
    smart_ptr5->name("Image Super-Resolution");
    UniquePtr<Test> smart_ptr6;
    smart_ptr6 = std::move(smart_ptr5);
    std::cout << "the name is: " << smart_ptr6->name() << std::endl;

    /* Step 4. C++11 shared_ptr 共享智能指针,
    Linux 引用计数的引入和技巧
    --------------------------------------- */
    std::cout << "==================================\n";
    SharedPtr<Test> smart_ptr7(new Test());

    smart_ptr7->name("JX");
    std::cout << "the name is: " << smart_ptr7->name() << std::endl;
    std::cout << "the name is: " << smart_ptr7.use_count() << std::endl;

    SharedPtr<Test> smart_ptr8;
    smart_ptr8 = smart_ptr7;
    std::cout << "the name is: " << (*smart_ptr8).name() << std::endl;
    std::cout << "the name is: " << smart_ptr8.use_count() << std::endl;

    /* Step 5. C++11 shared_ptr 循环引用的问题
    ------------------------------------------ */
    {
        std::cout << "---------------------------\n";
        std::shared_ptr<A> a_ptr(new A());
        std::shared_ptr<B> b_ptr(new B());

        std::cout << "the use count of a_ptr: " << a_ptr.use_count() << "\n";
        std::cout << "the use count of b_ptr: " << b_ptr.use_count() << "\n";

        a_ptr->m_b = b_ptr;
        b_ptr->m_a = a_ptr;

        std::cout << "the use count of a_ptr: " << a_ptr.use_count() << "\n";
        std::cout << "the use count of b_ptr: " << b_ptr.use_count() << "\n";
        // A and B 析构函数没有被调用, 资源无法释放, 内存泄露了
        // 所以提出了 weak_ptr,
        // 只需要将其中一个类(如B)的智能指针换成 weak_ptr 即可

        std::cout << "---------------------------\n";
    }

    /* Step 6. C++11 weak_ptr 辅助解决 shared_ptr 的循环引用问题
    ---------------------------------------------------------- */
    {
        std::cout << "---------------------------\n";
        SharedPtr<A_CUSTOM> a_ptr(new A_CUSTOM());
        SharedPtr<B_CUSTOM> b_ptr(new B_CUSTOM());

        std::cout << "the use count of a_ptr: " << a_ptr.use_count() << "\n";
        std::cout << "the use count of b_ptr: " << b_ptr.use_count() << "\n";

        a_ptr->m_b = b_ptr;
        b_ptr->m_a = a_ptr;

        std::cout << "the use count of a_ptr: " << a_ptr.use_count() << "\n";
        std::cout << "the use count of b_ptr: " << b_ptr.use_count() << "\n";

        std::cout << "---------------------------\n";
    }

    return 0;
}