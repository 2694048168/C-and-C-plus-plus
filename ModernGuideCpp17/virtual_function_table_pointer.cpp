/**
 * @file virtual_function_table_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

class base_no_virtual
{
public:
    void func1();
    void func2();

private:
    int m_age{42};

    std::string m_name{"weili"};
};

class base_virtual
{
public:
    void func1();
    void func2();

    virtual void play()
    {
        std::cout << "this is virtual function.\n";
    }
    // void *vptr; /* 虚函数表指针 */
    // vptr = &base_virtual::vftable;
    /* vtpr 指向 vtbl 虚函数表(虚函数的地址, 多态时调用) */

private:
    int m_age{42};

    std::string m_name{"weili"};
};

/**
 * @brief 虚函数/虚函数指针/虚函数表
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "class w/o virtual size: " << sizeof(base_no_virtual) << std::endl;
    std::cout << "class w/ virtual size: " << sizeof(base_virtual) << std::endl;

    base_no_virtual obj_no_virtual;
    base_virtual obj_virtual;
    std::cout << "obj. w/o virtual size: " << sizeof(obj_no_virtual) << std::endl;
    std::cout << "obj. w/ virtual size: " << sizeof(obj_virtual) << std::endl;

    return 0;
}