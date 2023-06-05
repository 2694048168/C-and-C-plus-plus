/**
 * @file smart_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

class Dog
{
public:
    Dog(const std::string_view &name)
        : m_name(name)
    {
        std::cout << "constructor of Dog: " << name << std::endl;
    }

    Dog() = default;

    ~Dog()
    {
        std::cout << "deconstructor of Dog: " << m_name << std::endl;
    }

    void dog_info() const
    {
        std::cout << "dog info name: " << m_name << std::endl;
    }

    std::string get_name() const
    {
        return m_name;
    }

    void set_dog_name(const std::string_view &name)
    {
        this->m_name = name;
    }

private:
    std::string m_name{"Mi"};
};

void pass_value_dog(std::unique_ptr<Dog> dog)
{
    dog->dog_info();
}

// void pass_reference_dog(std::unique_ptr<Dog> &dog)
void pass_reference_dog(const std::unique_ptr<Dog> &dog)
{
    dog->dog_info();
    // dog.reset(); /* non-const, 可以清空 */
}

std::unique_ptr<Dog> get_unique_ptr()
{
   std::unique_ptr<Dog> local_dog = std::make_unique<Dog>("Local Dog");
   std::cout << "local_dog address(get): " << local_dog.get() << std::endl;
   std::cout << "local_dog address(&): " << &local_dog << std::endl;
   return local_dog;
}

/**
 * @brief the smart pointer in Modern C++
 * raw pointer | unique_ptr | shared_ptr | weak_ptr
 * 
 * C++ 的指针包括两类
 * 1. 原始指针 (raw  pointer)
 * 2. 智能指针
 * 智能指针是原始指针的封装,会自动分配内存,尽量避免潜在的内存泄漏
 * 智能指针与 Rust 的内存安全机制？
 * 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ---------------------------------
    // stack
    Dog dog_1("dog_1");
    dog_1.dog_info();
    // scope
    {
        Dog dog_2("dog_2");
        dog_2.dog_info();
    }

    // ---------------------------------
    std::cout << "---------------------------------" << std::endl;
    // raw pointer
    Dog *dog_3 = new Dog("dog_3");
    dog_3->dog_info();
    {
        Dog *dog_4 = new Dog("dog_4");
        dog_4->dog_info();

        delete dog_4;
        dog_4 = nullptr;

        // how to detect the build-in type(deconstructor)?
        int *ptr = new int{42};
        // ! 如果不释放堆指针(heap), 问题很大啊 ^_^
        delete ptr;
        ptr = nullptr;
    }
    delete dog_3;
    dog_3 = nullptr;

    // ---------------------------------
    std::cout << "---------------------------------" << std::endl;
    std::unique_ptr<Dog> dog_5(new Dog("dog_5"));
    dog_5->dog_info();
    dog_5->set_dog_name("dog_5_setname");
    dog_5->dog_info();
    {
        std::unique_ptr<Dog> dog_6 = std::make_unique<Dog>();
        dog_6->dog_info();
        dog_6->set_dog_name("dog_6_setname");
        dog_6->dog_info();
    }
    std::cout << "dog_5 address: " << dog_5.get() << std::endl;

    // get and 解引用(dereference)
    // std::unique_ptr<int> u_ptr{new int{256}};
    std::unique_ptr<int> u_ptr = std::make_unique<int>(512);
    std::cout << "u_ptr address: " << u_ptr.get() << std::endl;
    std::cout << "u_ptr value: " << *u_ptr << std::endl;

    // ---------------------------------
    std::cout << "---------------------------------" << std::endl;
    /* std::unique_ptr 和函数调用, 不能 copy, 只能 move
        在作为函数参数或者返回值中一定要注意所有权.
    ----------------------------------------- */
    std::unique_ptr<Dog> dog_7 = std::make_unique<Dog>("dog_7");
    // pass_value_dog(dog_7);
    pass_value_dog(std::move(dog_7)); /* 所有权？ */
    // dog_7->dog_info(); /* runtime error */
    pass_value_dog(std::make_unique<Dog>()); /* 隐式转化 move */

    // pass by reference with non-const
    std::unique_ptr<Dog> dog_8 = std::make_unique<Dog>("dog_8");
    pass_reference_dog(dog_8);
    // dog_8->dog_info(); /* runtime error */
    std::cout << "dog_8 address: " << dog_8.get() << std::endl; /* reset() */

    // pass by reference with const
    std::unique_ptr<Dog> dog_9 = std::make_unique<Dog>("dog_9");
    pass_reference_dog(dog_9);
    std::cout << "dog_8 address: " << dog_8.get() << std::endl; /* reset() */
    dog_8->dog_info();

    // return std::unique_ptr 函数链式
    get_unique_ptr()->dog_info();

    // ---------------------------------
    std::cout << "---------------------------------" << std::endl;
    std::shared_ptr<int> shared_var1 = std::make_shared<int>(42);
    std::cout << "shared_ptr value1: " << *shared_var1 << std::endl;
    std::cout << "shared_ptr count: " << shared_var1.use_count() << std::endl;

    std::shared_ptr<int> shared_var2 = shared_var1; /* value copy */
    std::cout << "shared_ptr value2: " << *shared_var2 << std::endl;
    std::cout << "shared_ptr count: " << shared_var1.use_count() << std::endl;

    *shared_var2 = 256;
    std::cout << "shared_ptr value1: " << *shared_var1 << std::endl;
    std::cout << "shared_ptr value2: " << *shared_var2 << std::endl;

    std::shared_ptr<Dog> shared_dog_1 = std::make_shared<Dog>("shared_dog_1");
    std::cout << "shared_dog_1 count: " << shared_dog_1.use_count() << std::endl;
    shared_dog_1->dog_info();

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Hello Modern C++ for Smart Pointer.\n";

    return 0;
}