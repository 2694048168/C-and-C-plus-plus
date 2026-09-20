/**
 * @file 26_reference_counting.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-09-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <iostream>
#include <memory>

void func(std::shared_ptr<float> i)
{
    (*i)++;
}

struct Foo
{
    Foo()
    {
        std::cout << "Foo::Foo" << std::endl;
    }

    ~Foo()
    {
        std::cout << "Foo::~Foo" << std::endl;
    }

    void foo()
    {
        std::cout << "Foo::foo" << std::endl;
    }
};

void f(const Foo &)
{
    std::cout << "f(const Foo&)" << std::endl;
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    // auto pointer_f = new float(3.1415); // illegal, no direct assignment
    // Constructed a std::shared_ptr
    auto pointer_f = std::make_shared<float>(2.1415);
    func(pointer_f);
    std::cout << *pointer_f << std::endl; // 3.1415
    // The shared_ptr will be destructed before leaving the scope

    // -----------------------------------
    auto pointer  = std::make_shared<int>(10);
    auto pointer2 = pointer;                                                     // 引用计数+1
    auto pointer3 = pointer;                                                     // 引用计数+1
    int *p        = pointer.get();                                               // 这样不会增加引用计数
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   // 3
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; // 3
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; // 3

    pointer2.reset();
    std::cout << "reset pointer2:" << std::endl;
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   // 2
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; // pointer2 已 reset; 0
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; // 2
    pointer3.reset();
    std::cout << "reset pointer3:" << std::endl;
    std::cout << "pointer.use_count() = " << pointer.use_count() << std::endl;   // 1
    std::cout << "pointer2.use_count() = " << pointer2.use_count() << std::endl; // 0
    std::cout << "pointer3.use_count() = " << pointer3.use_count() << std::endl; // pointer3 已 reset; 0
    // -----------------------------------

    // -----------------------------------
    std::unique_ptr<Foo> p1(std::make_unique<Foo>());
    // p1 不空, 输出
    if (p1)
        p1->foo();
    {
        std::unique_ptr<Foo> p2(std::move(p1));
        // p2 不空, 输出
        f(*p2);
        // p2 不空, 输出
        if (p2)
            p2->foo();
        // p1 为空, 无输出
        if (p1)
            p1->foo();
        p1 = std::move(p2);
        // p2 为空, 无输出
        if (p2)
            p2->foo();
        std::cout << "p2 被销毁" << std::endl;
    }
    // p1 不空, 输出
    if (p1)
        p1->foo();
    // Foo 的实例会在离开作用域时被销毁
    // -----------------------------------

    return 0;
}

// clang++ 26_reference_counting.cpp -std=c++20 -o demo.exe
// g++ 26_reference_counting.cpp -std=c++20 -o demo.exe
