/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <memory>

struct SomeStruct
{
    int x;
    int y;
};

// ------------------------------------
int main(int argc, const char **argv)
{
    std::unique_ptr<SomeStruct> ptr{new SomeStruct()};
    ptr.reset(NULL);

    return 0;
}