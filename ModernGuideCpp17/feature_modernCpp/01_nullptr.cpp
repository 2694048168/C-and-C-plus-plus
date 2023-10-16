/**
 * @file 01_nullptr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

void call_function(int val)
{
    std::cout << "call function(int val)\n";
}

void call_function(char *val)
{
    std::cout << "call function(char* val)\n";
}

/**
 * @brief 源码是C++程序NULL就是0，如果是C程序NULL表示(void*)0;
 * 是由于 C++ 中, void * 类型无法隐式转换为其他类型的指针,
 * 此时使用 0 代替 ((void *)0), 用于解决空指针的问题,
 * 这个0（0x0000 0000）表示的就是虚拟地址空间中的0地址, 这块地址是只读的
 */
// ----------------------------
int main(int argc, char **argv)
{
    int   val = 42;
    // char* val_ptr = NULL;
    char *val_ptr = nullptr;

    call_function(val);
    call_function(val_ptr);

    call_function(nullptr);
    // call_function(NULL);

    return 0;
}
